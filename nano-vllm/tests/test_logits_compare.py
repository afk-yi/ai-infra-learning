"""Logits comparison test: nano-vllm vs HuggingFace reference.

Verifies that nano-vllm's forward pass produces logits numerically close to
HuggingFace transformers, proving weight loading and computation are correct.

Design
------
- **Standalone forward pass** (not LLM engine): bypasses the engine's built-in
  sampler to get raw logits for comparison.
- **Subprocess isolation**: each model comparison runs in a child process via
  ``subprocess.run``, so ``torch.distributed`` is guaranteed clean between runs.
  The child serializes results to a pickle tempfile; the parent reads and prints.
- **Both models loaded in the same process**: nano-vllm + HF reference side by
  side, same input_ids, same bfloat16 dtype. This eliminates tokenizer and dtype
  as confounding variables.
- **Prefill-only**: single-sequence prefill with minimal context (no KV cache
  needed). The empty k_cache/v_cache skip cache writes automatically.

Metrics
-------
- PASS/FAIL: ``torch.allclose(rtol=1e-2, atol=5.0)``. atol=5.0 accounts for
  bfloat16 flash-attention accumulation in deep/long models, while still
  catching catastrophic errors (e.g. Q/K-Norm bug was max_diff > 25).
- max_diff: largest per-logit absolute difference.
- mean_diff: average per-logit absolute difference (always < 0.1 for good runs).
- top10_overlap: how many of the top-10 predicted tokens match HF's top-10,
  averaged over all positions. 10.0 = perfect match; >= 9.0 = functionally
  equivalent.

Usage
-----
    conda activate nanovllm-test
    python tests/test_logits_compare.py          # both models
    python tests/test_logits_compare.py --model qwen3   # Qwen3 only
    python tests/test_logits_compare.py --model llama   # LLaMA only
"""
import json
import os
import pickle
import subprocess
import sys
import tempfile

import torch
import torch._dynamo
import torch.distributed as dist

torch._dynamo.config.disable = True
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from nanovllm.engine.model_runner import _load_model
from nanovllm.utils.context import reset_context, set_context
from nanovllm.utils.loader import load_model

QWEN3_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
LLAMA_PATH = os.path.expanduser("~/huggingface/Llama-3.1-8B-Instruct/")


def compare_logits(model_path, prompts):
    """Core: load both models, run forward pass on each prompt, compare logits."""
    # Clean up stale processes
    subprocess.run("kill $(lsof -t -i:2333) 2>/dev/null", shell=True)

    dist.init_process_group("nccl", "tcp://localhost:2333", world_size=1, rank=0)
    torch.cuda.set_device(0)

    # Load HF config shared by both
    config = AutoConfig.from_pretrained(model_path)

    # Load HF reference model (before setting default device, to avoid conflicts)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16
    ).cuda()
    hf_model.eval()

    # nano-vllm needs default device and dtype set before model creation
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)

    # Load nano-vllm model
    nv_model = _load_model(config)
    load_model(nv_model, model_path)
    nv_model.eval()

    # Tokenizer for encoding prompts
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for prompt in prompts:
        # Tokenize
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = enc.input_ids.cuda()
        N = input_ids.size(1)
        positions = torch.arange(N, dtype=torch.int64).unsqueeze(0)

        # --- nano-vllm forward ---
        cu_seqlens_q = torch.tensor([0, N], dtype=torch.int32)
        cu_seqlens_k = torch.tensor([0, N], dtype=torch.int32)
        slot_mapping = torch.full((N,), -1, dtype=torch.int32)
        set_context(True, cu_seqlens_q, cu_seqlens_k, N, N, slot_mapping)

        flat_ids = input_ids.flatten()
        flat_pos = positions.flatten()
        nv_hidden = nv_model(flat_ids, flat_pos)
        # compute_logits only returns last-token per seq; use F.linear for all positions
        nv_logits = torch.nn.functional.linear(nv_hidden, nv_model.lm_head.weight)
        reset_context()

        # --- HF forward ---
        with torch.inference_mode():
            hf_logits = hf_model(input_ids).logits.squeeze(0)  # [N, vocab]

        # --- Compare ---
        diff = (nv_logits.float() - hf_logits.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        # bfloat16 flash_attn vs eager_attn accumulates rounding errors with
        # sequence depth; atol=5.0 allows for this while catching real bugs
        # (e.g. the Q/K-Norm bug produced max_diff > 25).
        passed = torch.allclose(nv_logits, hf_logits, rtol=1e-2, atol=5.0)

        # Top-10 overlap per position (functional equivalence metric)
        top10_overlaps = []
        for i in range(N):
            nv_top10 = set(nv_logits[i].topk(10).indices.tolist())
            hf_top10 = set(hf_logits[i].topk(10).indices.tolist())
            top10_overlaps.append(len(nv_top10 & hf_top10))
        avg_overlap = sum(top10_overlaps) / len(top10_overlaps)

        results.append({
            "prompt": prompt if len(prompt) <= 60 else prompt[:57] + "...",
            "n_tokens": N,
            "passed": passed,
            "max_abs_diff": round(max_diff, 6),
            "mean_abs_diff": round(mean_diff, 6),
            "avg_top10_overlap": round(avg_overlap, 1),
        })

        del hf_logits, nv_logits, nv_hidden

    del nv_model, hf_model
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    return results


def run_compare(model_path, prompts):
    """Run compare_logits in a subprocess for clean dist isolation."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        output_file = f.name

    cmd = [
        "conda", "run", "-n", "nanovllm-test", "python",
        __file__, "subprocess", model_path, json.dumps(prompts), output_file,
    ]
    subprocess.run(cmd, check=True)

    with open(output_file, "rb") as f:
        results = pickle.load(f)
    os.unlink(output_file)
    return results


# ── Test cases ──────────────────────────────────────────────────────────────

SHORT_PROMPTS = [
    "Hello, how are you?",
    "The meaning of life is",
    "Python is a",
]

MEDIUM_PROMPTS = [
    "Explain the theory of relativity in simple terms. "
    "Einstein's theory can be understood by imagining you're on a train.",
]

LONG_PROMPTS = [
    ("Once upon a time, in a land far far away, there lived a wise old king "
     "who ruled over a vast and prosperous kingdom. The king had three sons, "
     "each brave and noble in their own way. " * 3),
]

# Prompts that include BOS token (as tokenizer would produce with add_special_tokens=True)
# A normal short prompt already gets BOS added by the tokenizer, so these are covered.


def _print_results(model_name, results):
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    all_pass = True
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        if not r["passed"]:
            all_pass = False
        print(f"  [{status}] {r['n_tokens']:4d}t  "
              f"max_diff={r['max_abs_diff']:.6f}  mean_diff={r['mean_abs_diff']:.6f}  "
              f"top10={r['avg_top10_overlap']:.1f}/10  {r['prompt']}")
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


def test_qwen3():
    """Qwen3 0.6B — control group, known to be correct."""
    prompts = SHORT_PROMPTS + MEDIUM_PROMPTS + LONG_PROMPTS
    results = run_compare(QWEN3_PATH, prompts)
    return _print_results("Qwen3-0.6B (control)", results)


def test_llama():
    """LLaMA 3.1 8B — primary target."""
    prompts = SHORT_PROMPTS + MEDIUM_PROMPTS + LONG_PROMPTS
    results = run_compare(LLAMA_PATH, prompts)
    return _print_results("LLaMA-3.1-8B (target)", results)


# ── Subprocess entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "subprocess":
        # Running inside subprocess — do actual comparison
        model_path = sys.argv[2]
        prompts = json.loads(sys.argv[3])
        output_file = sys.argv[4]
        results = compare_logits(model_path, prompts)
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
    else:
        # Main entry — run both tests
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", choices=["qwen3", "llama", "all"], default="all")
        args = parser.parse_args()

        ok = True
        if args.model in ("qwen3", "all"):
            ok &= test_qwen3()
        if args.model in ("llama", "all"):
            ok &= test_llama()

        sys.exit(0 if ok else 1)
