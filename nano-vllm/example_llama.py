"""LLaMA 3.1 8B example - test generation quality."""
import os
import sys
sys.path.insert(0, "/data/nano-vllm")

import torch
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Llama-3.1-8B-Instruct/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # Simple prompt - no chat template
    sampling_params = SamplingParams(temperature=0.3, max_tokens=64)
    prompts = ["The capital of France is"]
    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt!r}")
        print(f"Output: {output['text']!r}")
        print()

    # Chat template prompt
    sampling_params2 = SamplingParams(temperature=0.6, max_tokens=128)
    chat_prompts = [
        "What is the meaning of life?",
        "Explain gravity in simple terms.",
    ]
    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in chat_prompts
    ]
    outputs2 = llm.generate(chat_prompts, sampling_params2)
    for prompt, output in zip(chat_prompts, outputs2):
        print(f"Prompt: {prompt!r}")
        print(f"Output: {output['text']!r}")
        print()

    del llm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
