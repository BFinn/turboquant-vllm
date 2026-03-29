"""Tests for TurboQuant KV cache compression."""

import pytest
import torch

from turboquant import TurboQuantKVCache

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
