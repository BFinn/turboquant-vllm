"""
Inference test for TurboQuant vLLM plugin with Qwen3.5 35B A3B AWQ.

Runs the model with and without TurboQuant to compare:
1. Output quality (do we get coherent text?)
2. Memory usage (how much VRAM is the KV cache using?)
3. A simple needle-in-haystack retrieval test

Requires: vLLM installed, GPU with sufficient VRAM (CPU offloading enabled).
"""

import os
import sys
import time
import gc

# Configure TurboQuant BEFORE importing vllm
TURBOQUANT_ENABLED = os.environ.get("VLLM_TURBOQUANT_ENABLED", "1") != "0"

import torch


def get_gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def test_basic_generation(llm):
    """Test basic text generation quality."""
    from vllm import SamplingParams

    prompts = [
        "Explain quantum entanglement in simple terms:",
        "Write a Python function to find the longest common subsequence:",
        "What are the key differences between TCP and UDP?",
    ]

    params = SamplingParams(
        temperature=0.7,
        max_tokens=200,
        top_p=0.9,
    )

    print("\n=== Basic Generation Test ===")
    outputs = llm.generate(prompts, params)
    for output in outputs:
        prompt = output.prompt[:60] + "..."
        text = output.outputs[0].text[:200]
        print(f"\nPrompt: {prompt}")
        print(f"Output: {text}")
        print(f"Tokens: {len(output.outputs[0].token_ids)}")


def test_needle_in_haystack(llm, context_tokens=4096):
    """Hide a fact in filler text and see if the model retrieves it."""
    from vllm import SamplingParams

    needle = "The secret password for Project Zenith is: AURORA-7749-DELTA."

    filler = (
        "The quarterly report discussed various operational metrics including "
        "supply chain efficiency, customer retention rates, and regional "
        "expansion plans. The committee noted improvements in several areas "
        "while highlighting ongoing challenges in others. "
    )

    # Build context with needle buried in the middle
    filler_tokens_approx = len(filler.split()) * 1.3  # rough token estimate
    n_fillers = max(1, int(context_tokens / filler_tokens_approx))
    needle_pos = n_fillers // 2

    parts = []
    for i in range(n_fillers):
        if i == needle_pos:
            parts.append(f"\n[CONFIDENTIAL MEMO] {needle}\n\n")
        parts.append(filler)

    context = "".join(parts)
    prompt = f"{context}\n\nQuestion: What is the secret password for Project Zenith? Answer concisely:"

    params = SamplingParams(
        temperature=0.0,  # greedy for deterministic retrieval
        max_tokens=50,
    )

    print(f"\n=== Needle-in-Haystack ({context_tokens} token context) ===")
    print(f"Needle placed at position ~{needle_pos}/{n_fillers}")

    outputs = llm.generate([prompt], params)
    answer = outputs[0].outputs[0].text
    print(f"Answer: {answer}")

    found = "AURORA-7749-DELTA" in answer or "AURORA" in answer
    print(f"Needle retrieved: {'YES' if found else 'NO'}")
    return found


def test_memory_usage(llm):
    """Report GPU memory after loading."""
    print("\n=== Memory Usage ===")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  GPU allocated: {allocated:.2f} GB")
        print(f"  GPU reserved:  {reserved:.2f} GB")
    else:
        print("  No CUDA available")


def main():
    from vllm import LLM, SamplingParams

    model_name = os.environ.get(
        "TURBOQUANT_MODEL",
        "Qwen/Qwen3.5-35B-A3B-AWQ",
    )
    cpu_offload_gb = int(os.environ.get("TURBOQUANT_CPU_OFFLOAD_GB", "0"))
    max_model_len = int(os.environ.get("TURBOQUANT_MAX_MODEL_LEN", "8192"))

    tq_status = "ENABLED" if TURBOQUANT_ENABLED else "DISABLED"
    print(f"TurboQuant: {tq_status}")
    print(f"Model: {model_name}")
    print(f"CPU offload: {cpu_offload_gb} GB")
    print(f"Max model len: {max_model_len}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print("=" * 60)

    print("\nLoading model...")
    t0 = time.time()

    llm = LLM(
        model=model_name,
        enforce_eager=True,
        gpu_memory_utilization=0.90,
        cpu_offload_gb=cpu_offload_gb,
        max_model_len=max_model_len,
        max_num_seqs=2,
        trust_remote_code=True,
    )

    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    test_memory_usage(llm)
    test_basic_generation(llm)

    # Needle tests at increasing context
    for ctx in [2048, 4096]:
        if ctx <= max_model_len:
            test_needle_in_haystack(llm, ctx)

    print("\n" + "=" * 60)
    print("DONE")
    test_memory_usage(llm)


if __name__ == "__main__":
    main()
