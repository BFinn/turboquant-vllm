#!/usr/bin/env python3
"""Benchmark TurboQuant decode throughput via the vLLM API.

Runs multiple trials at different context lengths to isolate decode speed.
Reports median tok/s to reduce noise from MoE offloading variance.
"""

import json
import statistics
import sys
import time

import requests

URL = "http://127.0.0.1:8082/v1/completions"
MODEL = "Qwen3.5-35B-A3B"


def wait_for_server(timeout: int = 300) -> bool:
    """Wait for vLLM to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get("http://127.0.0.1:8082/v1/models", timeout=5)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(5)
    return False


def bench_decode(prompt: str, max_tokens: int = 300, trials: int = 3) -> dict:
    """Run decode benchmark, return stats."""
    results = []
    for t in range(trials):
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        start = time.time()
        resp = requests.post(URL, json=payload)
        elapsed = time.time() - start
        data = resp.json()
        usage = data.get("usage", {})
        comp = usage.get("completion_tokens", 0)
        prompt_toks = usage.get("prompt_tokens", 0)
        tok_s = comp / elapsed if elapsed > 0 else 0
        results.append({"prompt_tokens": prompt_toks, "completion_tokens": comp,
                        "elapsed": elapsed, "tok_s": tok_s})
        print(f"  trial {t+1}: {comp} tokens in {elapsed:.2f}s = {tok_s:.1f} tok/s")

    tok_s_vals = [r["tok_s"] for r in results]
    return {
        "median_tok_s": statistics.median(tok_s_vals),
        "mean_tok_s": statistics.mean(tok_s_vals),
        "min_tok_s": min(tok_s_vals),
        "max_tok_s": max(tok_s_vals),
        "trials": results,
    }


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {label}")
    print(f"{'='*60}\n")

    if not wait_for_server(timeout=10):
        print("Server not ready, waiting up to 5 minutes...")
        if not wait_for_server(timeout=300):
            print("ERROR: Server not reachable")
            sys.exit(1)

    # Short context (few blocks) - isolates per-token overhead
    print("--- Short context (13 prompt tokens, 300 output) ---")
    short = bench_decode(
        "List the first 100 prime numbers separated by commas:",
        max_tokens=300, trials=3,
    )

    # Medium context (~500 tokens prompt) - tests gather overhead
    filler = "The quick brown fox jumps over the lazy dog. " * 60
    print("\n--- Medium context (~500 prompt tokens, 300 output) ---")
    medium = bench_decode(
        filler + "\nNow list the first 100 prime numbers separated by commas:",
        max_tokens=300, trials=3,
    )

    print(f"\n{'='*60}")
    print(f"  RESULTS: {label}")
    print(f"{'='*60}")
    print(f"  Short context:  {short['median_tok_s']:.1f} tok/s (median)")
    print(f"  Medium context: {medium['median_tok_s']:.1f} tok/s (median)")
    print(f"{'='*60}\n")

    # Save results
    out = {"label": label, "short": short, "medium": medium,
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    fname = f"bench_results_{label}.json"
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {fname}")


if __name__ == "__main__":
    main()
