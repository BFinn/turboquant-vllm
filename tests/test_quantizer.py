"""Tests for MSE and inner product quantizers."""

import math

import pytest
import torch

from turboquant import TurboQuantMSE, TurboQuantProd

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
    bound = math.sqrt(3) * math.pi / 2 * (1 / (4**bits))
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
