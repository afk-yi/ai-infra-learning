"""LLaMA 3.1 8B benchmark."""
import os
import sys
sys.path.insert(0, "/data/nano-vllm")

import time
from random import randint, seed
from nanovllm import LLM, SamplingParams


def main():
    seed(0)
    num_seqs = 64
    max_input_len = 512
    max_output_len = 256

    path = os.path.expanduser("~/huggingface/Llama-3.1-8B-Instruct/")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len)) for _ in range(num_seqs)]

    # Warmup
    llm.generate(["Benchmark warmup: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"LLaMA 3.1 8B | Total: {total_tokens}tok | Time: {t:.2f}s | Throughput: {throughput:.2f}tok/s")

    del llm


if __name__ == "__main__":
    main()
