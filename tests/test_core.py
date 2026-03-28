"""Pytest tests for core TurboQuant quantization library.

Tests MSE distortion bounds, inner product accuracy, compression ratios,
and needle-in-haystack retrieval against theoretical predictions.
"""

import math

import pytest
import torch

from turboquant import TurboQuantMSE, TurboQuantProd, TurboQuantKVCache
from lloyd_max import LloydMaxCodebook


# ---------------------------------------------------------------------------
# Codebook
# ---------------------------------------------------------------------------

CODEBOOK_PARAMS = [(d, b) for d in [64, 128, 256] for b in [1, 2, 3, 4]]


@pytest.mark.parametrize("d,bits", CODEBOOK_PARAMS)
def test_codebook_level_count(d, bits):
    cb = LloydMaxCodebook(d, bits)
    assert cb.n_levels == 2 ** bits


@pytest.mark.parametrize("d,bits", [(128, 2), (128, 3), (256, 4)])
def test_codebook_symmetry(d, bits):
    cb = LloydMaxCodebook(d, bits)
    assert cb.centroids.sum().abs().item() < 0.01, "Centroids should be symmetric around 0"


# ---------------------------------------------------------------------------
# MSE Quantizer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_mse_distortion_within_bound(bits):
    d, n = 128, 1000
    q = TurboQuantMSE(d, bits, seed=42, device="cpu")
    x = torch.randn(n, d)
    x = x / x.norm(dim=-1, keepdim=True)
    x_hat, _ = q(x)
    mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()
    bound = math.sqrt(3) * math.pi / 2 * (1 / (4 ** bits))
    assert mse < bound * 1.5, f"MSE {mse:.6f} exceeds 1.5x theoretical bound {bound:.6f}"


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_mse_reconstruct_shape(bits):
    d, n = 128, 50
    q = TurboQuantMSE(d, bits, seed=42, device="cpu")
    x = torch.randn(n, d)
    x_hat, indices = q(x)
    assert x_hat.shape == (n, d)
    assert indices.shape[0] == n


# ---------------------------------------------------------------------------
# Inner Product (QJL)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [2, 3, 4])
def test_inner_product_unbiased(bits):
    d, n = 128, 2000
    q = TurboQuantProd(d, bits, seed=42, device="cpu")
    x = torch.randn(n, d)
    x = x / x.norm(dim=-1, keepdim=True)
    y = torch.randn(n, d)
    y = y / y.norm(dim=-1, keepdim=True)
    true_ip = (x * y).sum(dim=-1)
    compressed = q.quantize(x)
    est_ip = q.inner_product(y, compressed)
    bias = (est_ip - true_ip).mean().item()
    assert abs(bias) < 0.02, f"Bias {bias:.6f} too high for {bits}-bit"


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_inner_product_correlation(bits):
    d, n = 128, 2000
    q = TurboQuantProd(d, bits, seed=42, device="cpu")
    x = torch.randn(n, d)
    x = x / x.norm(dim=-1, keepdim=True)
    y = torch.randn(n, d)
    y = y / y.norm(dim=-1, keepdim=True)
    true_ip = (x * y).sum(dim=-1)
    compressed = q.quantize(x)
    est_ip = q.inner_product(y, compressed)
    corr = torch.corrcoef(torch.stack([true_ip, est_ip]))[0, 1].item()
    min_corr = {2: 0.75, 3: 0.88, 4: 0.95}[bits]
    assert corr > min_corr, f"Correlation {corr:.4f} below {min_corr} for {bits}-bit"


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [2, 3, 4])
def test_kv_cache_compression_ratio(bits):
    d_k, d_v, seq = 128, 128, 1024
    cache = TurboQuantKVCache(d_k, d_v, bits=bits, seed=42, device="cpu")
    cache.append(torch.randn(seq, d_k), torch.randn(seq, d_v))
    usage = cache.memory_usage_bits()
    assert usage["compression_ratio"] > 1.0, "Should compress"
    expected_min = {2: 6.0, 3: 4.0, 4: 3.0}[bits]
    assert usage["compression_ratio"] > expected_min, (
        f"{bits}-bit ratio {usage['compression_ratio']:.2f}x below {expected_min}x"
    )


def test_kv_cache_attention_shape():
    d = 128
    cache = TurboQuantKVCache(d, d, bits=3, seed=42, device="cpu")
    cache.append(torch.randn(512, d), torch.randn(512, d))
    scores = cache.attention_scores(torch.randn(1, d))
    assert scores.shape == (512,)


# ---------------------------------------------------------------------------
# Needle-in-Haystack Retrieval
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits,seq_len", [
    (2, 512), (2, 2048),
    (3, 512), (3, 2048), (3, 8192),
    (4, 512), (4, 2048), (4, 8192),
])
def test_needle_retrieval_exact(bits, seq_len):
    d = 128
    keys = torch.randn(seq_len, d)
    keys = keys / keys.norm(dim=-1, keepdim=True)
    needle_pos = seq_len // 3
    query = keys[needle_pos].clone().unsqueeze(0)
    q = TurboQuantProd(d, bits, seed=42, device="cpu")
    compressed = q.quantize(keys)
    est_ips = q.inner_product(query.expand(seq_len, -1), compressed)
    top5 = est_ips.topk(5).indices.tolist()
    assert needle_pos in top5, f"Needle {needle_pos} not in top-5: {top5}"


# ---------------------------------------------------------------------------
# GPU (skip if unavailable)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
def test_gpu_quantize_roundtrip():
    d, n, bits = 128, 1000, 3
    q = TurboQuantProd(d, bits, seed=42, device="cuda")
    x = torch.randn(n, d, device="cuda")
    x = x / x.norm(dim=-1, keepdim=True)
    compressed = q.quantize(x)
    x_hat = q.dequantize(compressed)
    assert x_hat.device.type == "cuda"
    cos = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).mean().item()
    assert cos > 0.85, f"GPU roundtrip cosine sim {cos:.4f} too low"
