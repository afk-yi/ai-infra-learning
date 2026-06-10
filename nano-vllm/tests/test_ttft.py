"""TTFT / decode-latency benchmark for mixed prefill/decode scheduling.

Measures scheduling latency for decode sequences under concurrent prefill load.
In the current all-or-nothing scheduler, decode sequences must wait for entire
prefill-only steps to complete, causing inter-token latency (ITL) spikes.
Mixed scheduling reduces this by running prefill and decode together.

Design
------
- **max_tokens > 1**: short requests enter decode phase after prefill. Their
  subsequent tokens are blocked during long prefill steps — this is what mixed
  scheduling fixes.
- **Monkey-patch postprocess**: records timestamps each time a sequence generates
  a token, giving per-token inter-token latency (ITL).
- **Short + long mix**: short requests (~2 tokens prompt, fast prefill) complete
  prefill in step 1 and enter running. Long requests (~2000 tokens, chunked
  prefill) create multi-step prefill phases that block decode.

Metrics
-------
- TTFT: time from submission to first token (from prefill postprocess).
  Same for baseline and mixed — first token comes from prefill completion.
- ITL (Inter-Token Latency): time between successive tokens for decode
  sequences. Spikes when a prefill-only step blocks decode.
- Total time: wall-clock time to complete all requests.
- Per-step composition: how many prefill vs decode tokens per step.

Usage
-----
    python tests/test_ttft.py                           # Qwen3 0.6B (default)
    python tests/test_ttft.py --model llama             # LLaMA 3.1 8B
    python tests/test_ttft.py --num-short 10 --num-long 2
    python tests/test_ttft.py --max-batched 1024 --long-len 4000
"""
import argparse
import os
import subprocess
import time
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams

QWEN3_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
LLAMA_PATH = os.path.expanduser("~/huggingface/Llama-3.1-8B-Instruct/")

# A natural paragraph, repeated to build long prompts.
_LONG_PARAGRAPH = (
    "Once upon a time, in a land far far away, there lived a wise old king "
    "who ruled over a vast and prosperous kingdom. The king had three sons, "
    "each brave and noble in their own way. The eldest son was a mighty warrior, "
    "the middle son a brilliant scholar, and the youngest a cunning diplomat. "
    "Together they defended the realm from all threats, both foreign and domestic. "
)


def build_long_prompt(tokenizer, target_tokens: int) -> str:
    """Build a prompt string that tokenizes to *roughly* target_tokens."""
    paragraph_tokens = len(tokenizer.encode(_LONG_PARAGRAPH))
    repeats = max(1, target_tokens // paragraph_tokens)
    return _LONG_PARAGRAPH * repeats


def run_ttft_benchmark(
    model_path: str,
    num_short: int,
    num_long: int,
    long_prompt_tokens: int,
    max_num_batched_tokens: int,
    max_tokens: int,
    verbose: bool = False,
) -> dict:
    """Run a single benchmark and return timing data.

    Monkey-patches ``Scheduler.postprocess`` to record per-token timestamps
    for decode sequences.

    Returns dict with:
        token_events: list of (seq_id, timestamp, token_index) — each time a
                      sequence generates a token. token_index 0 = first token.
        short_seq_ids: set[int]
        long_seq_ids: set[int]
        total_time: float
        num_steps: int
    """
    # Clean up stale NCCL port from prior crashed runs
    subprocess.run("kill $(lsof -t -i:2333) 2>/dev/null", shell=True)

    llm = LLM(
        model_path,
        enforce_eager=True,
        max_num_batched_tokens=max_num_batched_tokens,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    short_prompt = "Hello!"
    long_prompt = build_long_prompt(tokenizer, long_prompt_tokens)

    assert len(tokenizer.encode(short_prompt)) <= 10, "short prompt too long"
    actual_long_tokens = len(tokenizer.encode(long_prompt))
    print(f"  Short prompt: {len(tokenizer.encode(short_prompt))} tokens")
    print(f"  Long prompt:  {actual_long_tokens} tokens (target: {long_prompt_tokens})")

    # ── Monkey-patch postprocess to record per-token timestamps ──
    token_events: list[tuple[int, float, int]] = []  # (seq_id, timestamp, token_idx)

    original_postprocess = llm.scheduler.postprocess

    def timed_postprocess(seqs, token_ids):
        t = time.perf_counter()
        for seq in list(seqs):
            # postprocess will do: num_cached_tokens += num_scheduled_tokens
            # then check per-seq: if seq.is_prefill and num_cached_tokens < num_tokens → skip
            cached_after = seq.num_cached_tokens + seq.num_scheduled_tokens
            skip = seq.is_prefill and cached_after < seq.num_tokens
            if not skip:
                token_idx = seq.num_completion_tokens  # 0-based, before append
                token_events.append((seq.seq_id, t, token_idx))
        return original_postprocess(seqs, token_ids)

    llm.scheduler.postprocess = timed_postprocess

    # ── Submit all requests ──
    for _ in range(num_short):
        llm.add_request(short_prompt, SamplingParams(max_tokens=max_tokens))
    for _ in range(num_long):
        llm.add_request(long_prompt, SamplingParams(max_tokens=max_tokens))

    t_submit = time.perf_counter()

    # Classify seqs
    short_seq_ids: set[int] = set()
    long_seq_ids: set[int] = set()
    for _, _, s in llm.scheduler.waiting:
        if s.num_prompt_tokens <= 10:
            short_seq_ids.add(s.seq_id)
        else:
            long_seq_ids.add(s.seq_id)

    print(f"  Short seq_ids: {sorted(short_seq_ids)}")
    print(f"  Long seq_ids:  {sorted(long_seq_ids)}")

    # ── Run steps ──
    t_start = time.perf_counter()
    num_steps = 0

    while not llm.is_finished():
        output, num_tokens, has_prefill = llm.step()
        num_steps += 1
        if verbose:
            t_now = time.perf_counter()
            n_finished = len(output)
            n_waiting = len(llm.scheduler.waiting)
            n_running = len(llm.scheduler.running)
            step_type = "mixed" if has_prefill else "decode"
            print(f"  step {num_steps:2d}: {n_finished:2d} finished, "
                  f"waiting={n_waiting}, running={n_running}, "
                  f"{step_type}({num_tokens} tok), elapsed={t_now - t_start:.3f}s")

    t_end = time.perf_counter()
    total_time = t_end - t_start

    del llm
    torch.cuda.empty_cache()

    return {
        "token_events": token_events,
        "short_seq_ids": short_seq_ids,
        "long_seq_ids": long_seq_ids,
        "total_time": total_time,
        "num_steps": num_steps,
        "t_submit": t_submit,
    }


def compute_metrics(data: dict):
    """Compute TTFT, ITL, and completion-time statistics from token events."""
    events = data["token_events"]
    t_submit = data["t_submit"]
    short_ids = data["short_seq_ids"]
    long_ids = data["long_seq_ids"]

    # Group events by seq_id
    by_seq: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for seq_id, ts, tok_idx in events:
        by_seq[seq_id].append((ts, tok_idx))

    def compute_itl(seq_events):
        """Inter-token latencies for a sequence (sorted by token index)."""
        sorted_events = sorted(seq_events, key=lambda x: x[1])
        itls = []
        for i in range(1, len(sorted_events)):
            gap = sorted_events[i][0] - sorted_events[i - 1][0]
            itls.append(gap)
        # Also compute TTFT (first token time - submission)
        ttft = sorted_events[0][0] - t_submit if sorted_events else float('nan')
        return ttft, itls

    short_ttfts = []
    short_itls = []
    short_completion = []  # time to last token
    long_ttfts = []
    long_itls = []
    long_completion = []

    for seq_id, evts in by_seq.items():
        ttft, itls = compute_itl(evts)
        last_ts = max(ts for ts, _ in evts)
        completion = last_ts - t_submit

        if seq_id in short_ids:
            short_ttfts.append(ttft)
            short_itls.extend(itls)
            short_completion.append(completion)
        elif seq_id in long_ids:
            long_ttfts.append(ttft)
            long_itls.extend(itls)
            long_completion.append(completion)

    return {
        "short_ttft": short_ttfts,
        "short_itl": short_itls,
        "short_completion": short_completion,
        "long_ttft": long_ttfts,
        "long_itl": long_itls,
        "long_completion": long_completion,
        "total_time": data["total_time"],
        "num_steps": data["num_steps"],
    }


def p(arr, pct):
    """Percentile helper. Returns NaN for empty array."""
    if len(arr) == 0:
        return float('nan')
    return np.percentile(arr, pct)


def format_ms(arr):
    if len(arr) == 0:
        return "n/a"
    a = np.array(arr) * 1000
    return (f"min={a.min():.1f}ms  mean={a.mean():.1f}ms  "
            f"p50={p(a, 50):.1f}ms  p95={p(a, 95):.1f}ms  "
            f"p99={p(a, 99):.1f}ms  max={a.max():.1f}ms")


def print_report(m: dict, label: str):
    """Print a formatted latency report."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total time: {m['total_time']:.2f}s  |  Steps: {m['num_steps']}")
    print(f"  Short requests (n={len(m['short_ttft'])}):")
    print(f"    TTFT:       {format_ms(m['short_ttft'])}")
    print(f"    ITL:        {format_ms(m['short_itl'])}")
    print(f"    Completion: {format_ms(m['short_completion'])}")
    print(f"  Long requests (n={len(m['long_ttft'])}):")
    print(f"    TTFT:       {format_ms(m['long_ttft'])}")
    print(f"    ITL:        {format_ms(m['long_itl'])}")
    print(f"    Completion: {format_ms(m['long_completion'])}")
    if len(m['short_itl']) > 0:
        s = np.array(m['short_itl']) * 1000
        print(f"\n  Short ITL distribution: p50={p(s, 50):.1f}ms  "
              f"p95={p(s, 95):.1f}ms  p99={p(s, 99):.1f}ms")


def main():
    parser = argparse.ArgumentParser(
        description="TTFT / decode-latency benchmark for mixed scheduling",
    )
    parser.add_argument(
        "--model", choices=["qwen3", "llama"], default="qwen3",
        help="Model to benchmark (default: qwen3)",
    )
    parser.add_argument(
        "--num-short", type=int, default=10,
        help="Number of short-prompt requests (default: 10)",
    )
    parser.add_argument(
        "--num-long", type=int, default=1,
        help="Number of long-prompt requests (default: 1)",
    )
    parser.add_argument(
        "--long-len", type=int, default=2000,
        help="Target token count for long prompts (default: 2000)",
    )
    parser.add_argument(
        "--max-batched", type=int, default=512,
        help="max_num_batched_tokens — smaller values force chunked prefill "
             "(default: 512)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=20,
        help="max_tokens per request. Use >1 so short requests enter decode "
             "phase and experience blocking (default: 20)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-step schedule composition",
    )
    args = parser.parse_args()

    model_path = QWEN3_PATH if args.model == "qwen3" else LLAMA_PATH
    model_name = os.path.basename(os.path.dirname(model_path))

    print(f"Model: {model_name}")
    print(f"Short requests: {args.num_short}  |  Long requests: {args.num_long}")
    print(f"Long prompt target: {args.long_len} tokens  |  "
          f"max_num_batched_tokens: {args.max_batched}  |  "
          f"max_tokens: {args.max_tokens}")

    data = run_ttft_benchmark(
        model_path=model_path,
        num_short=args.num_short,
        num_long=args.num_long,
        long_prompt_tokens=args.long_len,
        max_num_batched_tokens=args.max_batched,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
    )

    metrics = compute_metrics(data)
    print_report(metrics, f"{model_name} (baseline)")


if __name__ == "__main__":
    main()
