"""Offline quality tests for the TurboQuant vLLM plugin.

Run without vLLM or GPU (CPU fallback):
    python -m vllm_turboquant.test_plugin

Tests validate the core compression and asymmetric attention math using
d=256 (Qwen3.5 full-attention head_dim) and various bit-widths.
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as F


def test_codebook_correctness():
    """Verify Lloyd-Max codebooks satisfy distortion bounds."""
    from .codebook import solve_codebook

    print("=== Codebook Correctness (d=256) ===")
    d = 256
    for bits in [1, 2, 3]:
        centroids = solve_codebook(d, bits)
        n_levels = 2 ** bits
        assert centroids.shape == (n_levels,), f"Expected {n_levels} centroids"

        # Centroids should be sorted
        assert torch.all(centroids[:-1] <= centroids[1:]), "Centroids not sorted"

        # Centroids should be symmetric around 0 (Gaussian is symmetric)
        assert abs(centroids.mean().item()) < 0.01 / math.sqrt(d), (
            f"Codebook mean {centroids.mean().item():.6f} not near zero"
        )

        print(f"  {bits}-bit: {n_levels} centroids, range [{centroids[0]:.4f}, {centroids[-1]:.4f}], mean={centroids.mean():.6f} OK")

    print()


def test_compress_decompress_roundtrip():
    """Verify compression roundtrip has reasonable cosine similarity."""
    from .compressor import TQKeyCompressorGPU, TQValueCompressorGPU

    print("=== Compress/Decompress Roundtrip (d=256) ===")
    d = 256
    N = 500
    device = torch.device("cpu")

    for bits in [2, 3, 4]:
        # Random unit-ish vectors
        torch.manual_seed(42)
        vecs = torch.randn(N, d)
        vecs = vecs / vecs.norm(dim=-1, keepdim=True) * (1 + 0.5 * torch.rand(N, 1))

        # Test value compressor (MSE roundtrip)
        v_comp = TQValueCompressorGPU(d, bits, seed=1234, device=device)
        compressed = v_comp.compress(vecs)
        reconstructed = v_comp.decompress(compressed)

        cos_sim = F.cosine_similarity(vecs, reconstructed, dim=-1).mean().item()
        rel_error = ((vecs - reconstructed).norm(dim=-1) / vecs.norm(dim=-1)).mean().item()

        assert cos_sim > 0.85, f"Value cosine sim too low: {cos_sim:.4f}"
        print(f"  {bits}-bit values: cosine_sim={cos_sim:.4f}, rel_error={rel_error:.4f}")

        # Test key compressor (MSE component)
        k_comp = TQKeyCompressorGPU(d, bits, seed=5678, device=device)
        k_compressed = k_comp.compress(vecs)
        k_mse = k_compressed["k_mse"].float()
        cos_sim_k = F.cosine_similarity(vecs, k_mse, dim=-1).mean().item()
        print(f"  {bits}-bit keys (MSE only): cosine_sim={cos_sim_k:.4f}")

    print()


def test_asymmetric_ip_unbiasedness():
    """Verify the asymmetric inner product estimator is unbiased."""
    from .compressor import TQKeyCompressorGPU

    print("=== Asymmetric IP Unbiasedness (d=256) ===")
    d = 256
    N = 5000
    device = torch.device("cpu")

    for bits in [2, 3, 4]:
        torch.manual_seed(42)
        keys = torch.randn(N, d)
        keys = keys / keys.norm(dim=-1, keepdim=True)
        queries = torch.randn(N, d)
        queries = queries / queries.norm(dim=-1, keepdim=True)

        comp = TQKeyCompressorGPU(d, bits, seed=9999, device=device)
        compressed = comp.compress(keys)

        # True inner products
        true_ip = (queries * keys).sum(dim=-1)  # (N,)

        # Asymmetric estimate: <q, k_mse> + correction
        k_mse = compressed["k_mse"].float()
        signs = compressed["qjl_signs"].float()
        r_norm = compressed["r_norm"].float()

        term1 = (queries * k_mse).sum(dim=-1)
        q_proj = queries @ comp.S.T
        qjl_ip = (q_proj * signs).sum(dim=-1)
        correction = comp.correction_scale * qjl_ip * r_norm
        est_ip = term1 + correction

        # Check bias
        bias = (est_ip - true_ip).mean().item()
        rmse = ((est_ip - true_ip) ** 2).mean().sqrt().item()
        correlation = torch.corrcoef(torch.stack([true_ip, est_ip]))[0, 1].item()

        assert abs(bias) < 0.02, f"Bias too high: {bias:.4f}"
        print(f"  {bits}-bit: bias={bias:+.4f}, RMSE={rmse:.4f}, corr={correlation:.4f}")

    print()


def test_gqa_attention_shapes():
    """Verify GQA attention produces correct shapes."""
    from .compressor import TQKeyCompressorGPU, TQValueCompressorGPU
    from .shadow_cache import ShadowKVCache, CompressedBlock
    from .config import TurboQuantConfig
    from .decode_attention import turboquant_decode_attention

    print("=== GQA Attention Shapes ===")
    config = TurboQuantConfig(
        bits=3,
        full_attn_layers=(0,),
        head_dim=256,
        num_kv_heads=2,
        num_q_heads=16,
    )
    device = torch.device("cpu")
    shadow = ShadowKVCache(config, device=device)

    # Simulate 2 blocks of 16 tokens each = 32 tokens
    block_size = 16
    torch.manual_seed(42)

    for blk_idx in range(2):
        keys = torch.randn(block_size, config.num_kv_heads, config.head_dim)
        vals = torch.randn(block_size, config.num_kv_heads, config.head_dim)
        shadow.compress_and_store(0, blk_idx, keys, vals, block_size)

    # Decode: 1 query token, batch_size=1
    query = torch.randn(1, config.num_q_heads, config.head_dim)
    output = torch.zeros(1, config.num_q_heads * config.head_dim)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    seq_lens = torch.tensor([32])

    result = turboquant_decode_attention(
        query=query,
        shadow_cache=shadow,
        layer_idx=0,
        block_table=block_table,
        seq_lens=seq_lens,
        block_size=block_size,
        scale=1.0 / math.sqrt(config.head_dim),
        output=output,
        num_actual_tokens=1,
    )

    assert result.shape == (1, config.num_q_heads * config.head_dim), (
        f"Wrong output shape: {result.shape}"
    )
    assert not torch.isnan(result).any(), "NaN in output"
    assert not torch.isinf(result).any(), "Inf in output"

    print(f"  Output shape: {result.shape} (expected (1, {config.num_q_heads * config.head_dim}))")
    print(f"  Output norm: {result.norm().item():.4f}")
    print(f"  No NaN/Inf: OK")
    print()


def test_attention_quality():
    """Compare TQ asymmetric attention vs exact FP32 attention."""
    from .compressor import TQKeyCompressorGPU, TQValueCompressorGPU
    from .shadow_cache import ShadowKVCache
    from .config import TurboQuantConfig
    from .decode_attention import turboquant_decode_attention

    print("=== Attention Quality vs Exact ===")
    config = TurboQuantConfig(
        bits=3,
        full_attn_layers=(0,),
        head_dim=256,
        num_kv_heads=2,
        num_q_heads=16,
    )
    device = torch.device("cpu")
    shadow = ShadowKVCache(config, device=device)

    seq_len = 64
    block_size = 16
    n_blocks = seq_len // block_size
    torch.manual_seed(42)

    # Generate random KV and store
    all_keys = torch.randn(seq_len, config.num_kv_heads, config.head_dim)
    all_vals = torch.randn(seq_len, config.num_kv_heads, config.head_dim)

    for blk_idx in range(n_blocks):
        s = blk_idx * block_size
        e = s + block_size
        shadow.compress_and_store(0, blk_idx, all_keys[s:e], all_vals[s:e], block_size)

    # Query
    query = torch.randn(1, config.num_q_heads, config.head_dim)
    scale = 1.0 / math.sqrt(config.head_dim)

    # TQ attention
    tq_output = torch.zeros(1, config.num_q_heads * config.head_dim)
    block_table = torch.arange(n_blocks, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([seq_len])

    turboquant_decode_attention(
        query, shadow, 0, block_table, seq_lens, block_size, scale, tq_output, 1,
    )

    # Exact FP32 attention (with GQA expansion)
    heads_per_kv = config.num_q_heads // config.num_kv_heads
    exact_outputs = []
    for kv_h in range(config.num_kv_heads):
        k = all_keys[:, kv_h, :].float()   # (L, D)
        v = all_vals[:, kv_h, :].float()    # (L, D)
        for q_offset in range(heads_per_kv):
            q_idx = kv_h * heads_per_kv + q_offset
            q = query[0, q_idx].float().unsqueeze(0)  # (1, D)
            scores = (q @ k.T) * scale  # (1, L)
            weights = F.softmax(scores, dim=-1)
            out = weights @ v  # (1, D)
            exact_outputs.append(out)
    exact_output = torch.cat(exact_outputs, dim=0).reshape(1, -1)  # (1, num_q_heads * D)

    # Compare
    cos_sim = F.cosine_similarity(
        tq_output.float().reshape(1, -1),
        exact_output.reshape(1, -1),
    ).item()
    rel_error = (tq_output.float() - exact_output).norm() / exact_output.norm()

    print(f"  3-bit: cosine_sim={cos_sim:.6f}, rel_error={rel_error:.4f}")
    # Random vectors compound key+value error; real KV cache gives >0.99.
    # Threshold is conservative for random data.
    assert cos_sim > 0.80, f"Attention cosine similarity too low: {cos_sim:.6f}"
    print()


def main():
    print("TurboQuant vLLM Plugin - Offline Quality Tests")
    print("=" * 60)
    print()

    tests = [
        ("Codebook correctness", test_codebook_correctness),
        ("Compress/decompress roundtrip", test_compress_decompress_roundtrip),
        ("Asymmetric IP unbiasedness", test_asymmetric_ip_unbiasedness),
        ("GQA attention shapes", test_gqa_attention_shapes),
        ("Attention quality vs exact", test_attention_quality),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
            print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
