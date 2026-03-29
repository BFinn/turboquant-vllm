"""Cached Lloyd-Max codebook solver for the vLLM plugin.

Delegates to turboquant.lloyd_max (the canonical solver) and caches results.
"""

from __future__ import annotations

import logging

import torch

from turboquant.lloyd_max import solve_lloyd_max

logger = logging.getLogger(__name__)

_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def get_codebook(d: int, bits: int) -> torch.Tensor:
    """Get cached codebook centroids for (d, bits). Thread-safe for reads after init."""
    key = (d, bits)
    if key not in _CACHE:
        logger.info("Solving Lloyd-Max codebook for d=%d, bits=%d", d, bits)
        centroids, _ = solve_lloyd_max(d, bits)
        _CACHE[key] = centroids
        logger.info("Codebook solved: %d centroids", 2**bits)
    return _CACHE[key]
