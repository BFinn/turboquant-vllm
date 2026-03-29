"""Tests for the Lloyd-Max codebook solver."""

import pytest

from turboquant.lloyd_max import LloydMaxCodebook

# ---------------------------------------------------------------------------
# Codebook
# ---------------------------------------------------------------------------

CODEBOOK_PARAMS = [(d, b) for d in [64, 128, 256] for b in [1, 2, 3, 4]]


@pytest.mark.parametrize("d,bits", CODEBOOK_PARAMS)
def test_codebook_level_count(d, bits):
    cb = LloydMaxCodebook(d, bits)
    assert cb.n_levels == 2**bits


@pytest.mark.parametrize("d,bits", [(128, 2), (128, 3), (256, 4)])
def test_codebook_symmetry(d, bits):
    cb = LloydMaxCodebook(d, bits)
    assert cb.centroids.sum().abs().item() < 0.01, "Centroids should be symmetric around 0"
