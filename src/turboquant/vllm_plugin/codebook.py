"""Cached Lloyd-Max codebook solver for the vLLM plugin.

Delegates to turboquant.lloyd_max (the canonical solver) and caches results
both in-memory and on disk (~/.cache/turboquant/) to avoid recomputation
across workers in tensor-parallel setups.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from turboquant.lloyd_max import solve_lloyd_max

logger = logging.getLogger(__name__)

_CACHE: dict[tuple[int, int], torch.Tensor] = {}
_DISK_CACHE_DIR = Path.home() / ".cache" / "turboquant" / "codebooks"


def _disk_path(d: int, bits: int) -> Path:
    return _DISK_CACHE_DIR / f"lloyd_max_d{d}_b{bits}.pt"


def get_codebook(d: int, bits: int) -> torch.Tensor:
    """Get cached codebook centroids for (d, bits). Thread-safe for reads after init."""
    key = (d, bits)
    if key in _CACHE:
        return _CACHE[key]

    # Try disk cache first
    disk_path = _disk_path(d, bits)
    if disk_path.exists():
        try:
            centroids = torch.load(disk_path, weights_only=True)
            logger.info("Loaded codebook from disk: d=%d, bits=%d", d, bits)
            _CACHE[key] = centroids
            return centroids
        except Exception:
            logger.warning("Corrupt codebook cache at %s, recomputing", disk_path)

    # Solve and cache to both memory and disk
    logger.info("Solving Lloyd-Max codebook for d=%d, bits=%d", d, bits)
    centroids, _ = solve_lloyd_max(d, bits)
    _CACHE[key] = centroids
    logger.info("Codebook solved: %d centroids", 2**bits)

    try:
        _DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(centroids, disk_path)
        logger.info("Codebook cached to %s", disk_path)
    except OSError:
        logger.warning("Could not write codebook cache to %s", disk_path)

    return centroids
