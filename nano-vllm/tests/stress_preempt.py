"""
Stress test: force KV cache exhaustion to trigger decode preemption.
Run with: python tests/stress_preempt.py
"""
import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams


def main():
    seed(0)
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    # Tight KV cache: enough for all prefills, NOT enough for decode
    llm = LLM(path, enforce_eager=False, max_model_len=4096,
              gpu_memory_utilization=0.55)

    # Long prompts → prefill uses many blocks, preempt recompute cost is HIGH
    # Long outputs → seqs stay in running, blocks run out during decode
    num_seqs = 32
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(1500, 2500))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True,
                       max_tokens=randint(300, 600))
        for _ in range(num_seqs)
    ]

    llm.generate(["warmup"], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, "
          f"Throughput: {total_tokens/t:.2f}tok/s")


if __name__ == "__main__":
    main()
