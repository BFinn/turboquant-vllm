"""Needle-in-haystack retrieval tests for TurboQuant."""

import pytest
import torch

from turboquant import TurboQuantProd

# ---------------------------------------------------------------------------
# Needle-in-Haystack Retrieval
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bits,seq_len",
    [
        (2, 512),
        (2, 2048),
        (3, 512),
        (3, 2048),
        (3, 8192),
        (4, 512),
        (4, 2048),
        (4, 8192),
    ],
)
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
