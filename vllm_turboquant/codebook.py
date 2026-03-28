"""Lloyd-Max codebook solver with global caching.

Solves once per (head_dim, bits) pair and caches the result. The codebook
is a small tensor (2^bits centroids) so memory is negligible.
"""

from __future__ import annotations

import math
import logging

import torch
from scipy import integrate

logger = logging.getLogger(__name__)

_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def _gaussian_pdf(x: float, d: int) -> float:
    """N(0, 1/d) approximation of the post-rotation coordinate distribution."""
    sigma2 = 1.0 / d
    return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(-x * x / (2 * sigma2))


def solve_codebook(d: int, bits: int, max_iter: int = 200, tol: float = 1e-10) -> torch.Tensor:
    """Solve Lloyd-Max optimal quantizer for Gaussian N(0, 1/d).

    Returns centroids tensor of shape (2^bits,) in float32.
    """
    n_levels = 2 ** bits
    sigma = 1.0 / math.sqrt(d)
    pdf = lambda x: _gaussian_pdf(x, d)

    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
        edges = [lo * 3] + boundaries + [hi * 3]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
            den, _ = integrate.quad(pdf, a, b)
            new_centroids.append(num / den if den > 1e-15 else centroids[i])
        if max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels)) < tol:
            break
        centroids = new_centroids

    return torch.tensor(centroids, dtype=torch.float32)


def get_codebook(d: int, bits: int) -> torch.Tensor:
    """Get cached codebook centroids for (d, bits). Thread-safe for reads after init."""
    key = (d, bits)
    if key not in _CACHE:
        logger.info("Solving Lloyd-Max codebook for d=%d, bits=%d", d, bits)
        _CACHE[key] = solve_codebook(d, bits)
        logger.info("Codebook solved: %d centroids", 2 ** bits)
    return _CACHE[key]
