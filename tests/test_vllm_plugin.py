"""Pytest tests for the vLLM TurboQuant plugin.

Validates codebook solver, compressor roundtrips, asymmetric inner products,
GQA attention shapes, and attention quality vs exact FP32. All CPU-only.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from vllm_turboquant.codebook import solve_codebook
from vllm_turboquant.compressor import TQKeyCompressorGPU, TQValueCompressorGPU
from vllm_turboquant.config import TurboQuantConfig
from vllm_turboquant.shadow_cache import ShadowKVCache
from vllm_turboquant.decode_attention import turboquant_decode_attention


# ---------------------------------------------------------------------------
# Codebook
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [1, 2, 3])
def test_codebook_sorted_and_symmetric(bits):
    d = 256
    centroids = solve_codebook(d, bits)
    assert centroids.shape == (2 ** bits,)
    assert torch.all(centroids[:-1] <= centroids[1:]), "Centroids not sorted"
    assert abs(centroids.mean().item()) < 0.01 / math.sqrt(d), (
        f"Codebook mean {centroids.mean():.6f} not near zero"
    )


# ---------------------------------------------------------------------------
# Value Compressor Roundtrip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [2, 3, 4])
def test_value_compress_decompress(bits):
    d, n = 256, 500
    torch.manual_seed(42)
    vecs = torch.randn(n, d)
    vecs = vecs / vecs.norm(dim=-1, keepdim=True) * (1 + 0.5 * torch.rand(n, 1))
    comp = TQValueCompressorGPU(d, bits, seed=1234, device=torch.device("cpu"))
    compressed = comp.compress(vecs)
    reconstructed = comp.decompress(compressed)
    cos_sim = F.cosine_similarity(vecs, reconstructed, dim=-1).mean().item()
    assert cos_sim > 0.85, f"Value cosine sim {cos_sim:.4f} too low at {bits}-bit"


# ---------------------------------------------------------------------------
# Key Compressor Roundtrip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [2, 3, 4])
def test_key_compress_cosine(bits):
    d, n = 256, 500
    torch.manual_seed(42)
    vecs = torch.randn(n, d)
    vecs = vecs / vecs.norm(dim=-1, keepdim=True)
    comp = TQKeyCompressorGPU(d, bits, seed=5678, device=torch.device("cpu"))
    compressed = comp.compress(vecs)
    cos_sim = F.cosine_similarity(vecs, compressed["k_mse"].float(), dim=-1).mean().item()
    assert cos_sim > 0.75, f"Key MSE cosine sim {cos_sim:.4f} too low at {bits}-bit"


# ---------------------------------------------------------------------------
# Asymmetric Inner Product Unbiasedness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [2, 3, 4])
def test_asymmetric_ip_unbiased(bits):
    d, n = 256, 5000
    torch.manual_seed(42)
    keys = torch.randn(n, d)
    keys = keys / keys.norm(dim=-1, keepdim=True)
    queries = torch.randn(n, d)
    queries = queries / queries.norm(dim=-1, keepdim=True)
    comp = TQKeyCompressorGPU(d, bits, seed=9999, device=torch.device("cpu"))
    compressed = comp.compress(keys)
    true_ip = (queries * keys).sum(dim=-1)
    k_mse = compressed["k_mse"].float()
    signs = compressed["qjl_signs"].float()
    r_norm = compressed["r_norm"].float()
    term1 = (queries * k_mse).sum(dim=-1)
    q_proj = queries @ comp.S.T
    qjl_ip = (q_proj * signs).sum(dim=-1)
    correction = comp.correction_scale * qjl_ip * r_norm
    est_ip = term1 + correction
    bias = (est_ip - true_ip).mean().item()
    assert abs(bias) < 0.02, f"Bias {bias:.4f} too high at {bits}-bit"


# ---------------------------------------------------------------------------
# GQA Attention Shapes
# ---------------------------------------------------------------------------

def _make_shadow(config, seq_len=32, block_size=16):
    device = torch.device("cpu")
    shadow = ShadowKVCache(config, device=device)
    torch.manual_seed(42)
    n_blocks = seq_len // block_size
    for blk in range(n_blocks):
        keys = torch.randn(block_size, config.num_kv_heads, config.head_dim)
        vals = torch.randn(block_size, config.num_kv_heads, config.head_dim)
        shadow.compress_and_store(0, blk, keys, vals, block_size)
    return shadow, n_blocks


def test_gqa_output_shape():
    config = TurboQuantConfig(
        bits=3, full_attn_layers=(0,), head_dim=256, num_kv_heads=2, num_q_heads=16,
    )
    shadow, n_blocks = _make_shadow(config)
    query = torch.randn(1, config.num_q_heads, config.head_dim)
    output = torch.zeros(1, config.num_q_heads * config.head_dim)
    block_table = torch.arange(n_blocks, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([32])
    result = turboquant_decode_attention(
        query, shadow, 0, block_table, seq_lens, 16,
        1.0 / math.sqrt(config.head_dim), output, 1,
    )
    assert result.shape == (1, config.num_q_heads * config.head_dim)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


# ---------------------------------------------------------------------------
# Attention Quality vs Exact FP32
# ---------------------------------------------------------------------------

def test_attention_quality_vs_exact():
    config = TurboQuantConfig(
        bits=3, full_attn_layers=(0,), head_dim=256, num_kv_heads=2, num_q_heads=16,
    )
    device = torch.device("cpu")
    shadow = ShadowKVCache(config, device=device)
    seq_len, block_size = 64, 16
    n_blocks = seq_len // block_size
    torch.manual_seed(42)
    all_keys = torch.randn(seq_len, config.num_kv_heads, config.head_dim)
    all_vals = torch.randn(seq_len, config.num_kv_heads, config.head_dim)
    for blk in range(n_blocks):
        s, e = blk * block_size, (blk + 1) * block_size
        shadow.compress_and_store(0, blk, all_keys[s:e], all_vals[s:e], block_size)

    query = torch.randn(1, config.num_q_heads, config.head_dim)
    scale = 1.0 / math.sqrt(config.head_dim)

    # TQ attention
    tq_out = torch.zeros(1, config.num_q_heads * config.head_dim)
    block_table = torch.arange(n_blocks, dtype=torch.int32).unsqueeze(0)
    turboquant_decode_attention(
        query, shadow, 0, block_table, torch.tensor([seq_len]), block_size,
        scale, tq_out, 1,
    )

    # Exact FP32 attention
    heads_per_kv = config.num_q_heads // config.num_kv_heads
    exact_parts = []
    for kv_h in range(config.num_kv_heads):
        k = all_keys[:, kv_h].float()
        v = all_vals[:, kv_h].float()
        for q_off in range(heads_per_kv):
            q = query[0, kv_h * heads_per_kv + q_off].float().unsqueeze(0)
            w = F.softmax((q @ k.T) * scale, dim=-1)
            exact_parts.append(w @ v)
    exact_out = torch.cat(exact_parts, dim=0).reshape(1, -1)

    cos_sim = F.cosine_similarity(tq_out.float().reshape(1, -1), exact_out.reshape(1, -1)).item()
    assert cos_sim > 0.80, f"Attention cosine sim {cos_sim:.6f} too low"
