"""
End-to-end correctness test: verify partial preemption produces identical
output as no-preemption baseline.

Run with: python tests/test_preempt_correctness.py

Determinism requirements:
  - enforce_eager=True (no FlashAttention fused-kernel non-determinism)
  - Low temperature → effectively greedy (robust to any floating-point noise)
  - Same random seeds for both runs
  - Each run in its own subprocess for clean torch.distributed isolation
"""
import os
import pickle
import random
import subprocess
import sys
import tempfile
import torch
from nanovllm import LLM, SamplingParams


RUNNER_SCRIPT = """
import os, pickle, random, torch, sys

with open({input_path!r}, "rb") as f:
    data = pickle.load(f)

torch.manual_seed(data["seed"])
torch.cuda.manual_seed(data["seed"]) if torch.cuda.is_available() else None
random.seed(data["seed"])

from nanovllm import LLM, SamplingParams
from nanovllm.engine import block_manager as bm_module

_preempt_count = [0]
_orig_free_tail = bm_module.BlockManager.free_tail_blocks
def _counting_free_tail(self, seq, n):
    _preempt_count[0] += 1
    return _orig_free_tail(self, seq, n)
bm_module.BlockManager.free_tail_blocks = _counting_free_tail

path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
sampling_params = SamplingParams(**data["sampling_params"])

print(f"Loading model with gpu_memory_utilization={{data['gpu_memory_utilization']}}...", flush=True)
llm = LLM(path, enforce_eager=True, max_model_len=4096,
          gpu_memory_utilization=data["gpu_memory_utilization"])

torch.manual_seed(data["seed"])
torch.cuda.manual_seed(data["seed"]) if torch.cuda.is_available() else None

bm = llm.scheduler.block_manager
total_blocks = len(bm.blocks)
free0 = len(bm.free_block_ids)
print(f"KV blocks: {{total_blocks}} total, {{free0}} free (before generate)", flush=True)
if data.get("reserve_blocks"):
    bs = llm.scheduler.block_size
    needed = sum((len(p) + bs - 1) // bs for p in data["prompt_token_ids"])
    to_reserve = len(bm.free_block_ids) - needed
    if to_reserve > 0:
        for _ in range(to_reserve):
            bid = bm._allocate_block()
            bm.blocks[bid].ref_count = 999
        print(f"Reserved {{to_reserve}} blocks, {{len(bm.free_block_ids)}} free (prefill needs {{needed}})", flush=True)

outputs = llm.generate(data["prompt_token_ids"], sampling_params, use_tqdm=True)
token_ids = [o["token_ids"] for o in outputs]
print(f"Preemption events: {{_preempt_count[0]}}", flush=True)

with open({output_path!r}, "wb") as f:
    pickle.dump({{"token_ids": token_ids, "preempt_count": _preempt_count[0]}}, f)
print("Done", flush=True)
"""


def run_and_save(gpu_memory_utilization, prompt_token_ids, sampling_params, output_path, seed, label, reserve_blocks=False):
    """Run LLM.generate in a subprocess and save token outputs."""
    input_data = {
        "seed": seed,
        "gpu_memory_utilization": gpu_memory_utilization,
        "prompt_token_ids": prompt_token_ids,
        "sampling_params": {
            "temperature": sampling_params.temperature,
            "max_tokens": sampling_params.max_tokens,
            "ignore_eos": sampling_params.ignore_eos,
        },
        "reserve_blocks": reserve_blocks,
    }
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        input_path = f.name
        pickle.dump(input_data, f)

    try:
        code = RUNNER_SCRIPT.format(input_path=input_path, output_path=output_path)
        print(f"\n{'='*60}")
        print(f"Running: {label}")
        print(f"{'='*60}")
        result = subprocess.run([sys.executable, "-c", code], check=True)
        if result.returncode != 0:
            raise RuntimeError(f"Subprocess failed with code {result.returncode}")
    finally:
        os.unlink(input_path)


def main():
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)

    # Use 2 seqs with identical prompt length. After block reservation, prefill
    # uses all blocks, so decode triggers partial preempt (not self-preempt,
    # which has a known assert bug).
    num_seqs = 2
    prompt_token_ids = [[random.randint(0, 10000) for _ in range(1000)] for _ in range(num_seqs)]
    sampling_params = SamplingParams(temperature=1e-6, max_tokens=256, ignore_eos=True)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        golden_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        test_path = f.name

    try:
        # --- Baseline: large KV cache, no preemption ---
        run_and_save(
            gpu_memory_utilization=0.90,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            output_path=golden_path,
            seed=seed,
            label="Baseline: large KV cache (no preemption)",
        )

        # --- Test: tight KV cache, preemption expected ---
        run_and_save(
            gpu_memory_utilization=0.05,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            output_path=test_path,
            seed=seed,
            label="Test: tight KV cache (preemption expected)",
            reserve_blocks=True,
        )

        # --- Compare ---
        with open(golden_path, "rb") as f:
            golden_data = pickle.load(f)
        with open(test_path, "rb") as f:
            test_data = pickle.load(f)

        golden_tokens = golden_data["token_ids"]
        test_tokens = test_data["token_ids"]
        golden_preempt = golden_data["preempt_count"]
        test_preempt = test_data["preempt_count"]

        print()
        print(f"Preempt events: baseline={golden_preempt}, test={test_preempt}")
        if test_preempt == 0:
            print("WARNING: No preemption triggered in tight-cache run — both paths are identical.")
        print()
        print("=" * 60)
        print("Comparing outputs...")
        print("=" * 60)

        all_match = True
        for i in range(num_seqs):
            g = golden_tokens[i]
            t = test_tokens[i]
            if g == t:
                print(f"  seq[{i}] len(golden)={len(g)}, len(test)={len(t)}: MATCH")
            else:
                all_match = False
                print(f"  seq[{i}] len(golden)={len(g)}, len(test)={len(t)}: MISMATCH")
                for j in range(min(len(g), len(t))):
                    if g[j] != t[j]:
                        print(f"    First diff at pos {j}: golden={g[j]}, test={t[j]}")
                        print(f"    golden[{max(0,j-3)}:{j+3}]={g[max(0,j-3):j+3]}")
                        print(f"    test[{max(0,j-3)}:{j+3}]={t[max(0,j-3):j+3]}")
                        break
                if len(g) != len(t):
                    print(f"    Length diff: golden={len(g)}, test={len(t)}")

        print()
        if all_match:
            print("PASS: All outputs match. Partial preemption is empirically correct.")
        else:
            print("FAIL: Output mismatches detected.")
            sys.exit(1)

    finally:
        os.unlink(golden_path)
        os.unlink(test_path)


if __name__ == "__main__":
    main()
