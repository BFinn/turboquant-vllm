"""GPU-optimized TurboQuant compressors for vLLM integration.

Adapted from compressors.py TurboQuantCompressorV2 (keys) and
TurboQuantCompressorMSE (values). Each compressor owns its own rotation
matrix Pi, codebook centroids, and QJL projection matrix S.
"""

from __future__ import annotations

import math

import torch

from .codebook import get_codebook


def _make_rotation_matrix(d: int, seed: int, device: torch.device) -> torch.Tensor:
    """Haar-distributed random orthogonal matrix via QR of Gaussian."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device)


def _make_qjl_matrix(d: int, seed: int, device: torch.device) -> torch.Tensor:
    """Random Gaussian projection matrix for QJL (d x d)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    S = torch.randn(d, d, generator=gen)
    return S.to(device)


class TQKeyCompressorGPU:
    """Key compressor: (bits-1)-bit MSE + 1-bit QJL on residuals.

    Stores k_mse (fp16 reconstruction), qjl_signs (int8), and residual norms
    for the asymmetric inner product estimator.
    """

    def __init__(self, head_dim: int, bits: int, seed: int, device: torch.device):
        self.head_dim = head_dim
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.device = device

        self.Pi = _make_rotation_matrix(head_dim, seed, device)
        self.centroids = get_codebook(head_dim, self.mse_bits).to(device)
        self.S = _make_qjl_matrix(head_dim, seed + 10000, device)

        self.correction_scale = math.sqrt(math.pi / 2) / head_dim

    @torch.no_grad()
    def compress(self, keys: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compress key vectors.

        Args:
            keys: (N, head_dim) fp16/bf16 on GPU

        Returns dict:
            key_indices: (N, head_dim) uint8 - codebook indices for rotated unit vectors
            key_norms:   (N,) fp16 - original vector L2 norms
            qjl_signs:   (N, head_dim) int8 - {-1, +1} sign bits
            r_norm:      (N,) fp16 - residual L2 norms
        """
        flat = keys.float()

        # Normalize to unit vectors, preserving original norms
        vec_norms = torch.norm(flat, dim=-1, keepdim=True)  # (N, 1)
        flat_unit = flat / (vec_norms + 1e-8)

        # Rotate and quantize
        rotated = flat_unit @ self.Pi.T  # (N, D)
        diffs = rotated.unsqueeze(-1) - self.centroids  # (N, D, n_levels)
        indices = diffs.abs().argmin(dim=-1).to(torch.uint8)  # (N, D)

        # MSE reconstruction in original space (for residual computation only)
        reconstructed_rotated = self.centroids[indices.long()]  # (N, D)
        k_mse = (reconstructed_rotated @ self.Pi) * vec_norms  # (N, D)

        # Residual and QJL
        residual = flat - k_mse
        r_norm = torch.norm(residual, dim=-1)  # (N,)
        projected = residual @ self.S.T  # (N, D)
        signs = (projected >= 0).to(torch.int8) * 2 - 1  # {-1, +1} as int8

        return {
            "key_indices": indices,
            "key_norms": vec_norms.squeeze(-1).half(),
            "qjl_signs": signs,
            "r_norm": r_norm.half(),
        }

    @torch.no_grad()
    def reconstruct_k_mse(self, indices: torch.Tensor, norms: torch.Tensor,
                          dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Reconstruct k_mse from stored indices and norms.

        Args:
            indices: (N, head_dim) uint8 codebook indices
            norms:   (N,) fp16 original vector norms

        Returns: (N, head_dim) in requested dtype
        """
        reconstructed_rotated = self.centroids[indices.long()]  # (N, D)
        k_mse = (reconstructed_rotated @ self.Pi) * norms.float().unsqueeze(-1)
        return k_mse.to(dtype)


class TQValueCompressorGPU:
    """Value compressor: full-bits MSE (no QJL needed for values)."""

    def __init__(self, head_dim: int, bits: int, seed: int, device: torch.device):
        self.head_dim = head_dim
        self.bits = bits
        self.device = device

        self.Pi = _make_rotation_matrix(head_dim, seed, device)
        self.centroids = get_codebook(head_dim, bits).to(device)

    @torch.no_grad()
    def compress(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compress value vectors.

        Args:
            values: (N, head_dim) fp16/bf16 on GPU

        Returns dict:
            indices:  (N, head_dim) uint8 - codebook indices
            norms:    (N,) fp16 - original vector norms
        """
        flat = values.float()
        vec_norms = torch.norm(flat, dim=-1, keepdim=True)
        flat_unit = flat / (vec_norms + 1e-8)

        rotated = flat_unit @ self.Pi.T
        diffs = rotated.unsqueeze(-1) - self.centroids
        indices = diffs.abs().argmin(dim=-1).to(torch.uint8)

        return {
            "indices": indices,
            "norms": vec_norms.squeeze(-1).half(),
        }

    @torch.no_grad()
    def decompress(self, compressed: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decompress values back to fp16.

        Returns: (N, head_dim) fp16
        """
        indices = compressed["indices"].long()
        norms = compressed["norms"].float()
        reconstructed = self.centroids[indices] @ self.Pi  # (N, D)
        return (reconstructed * norms.unsqueeze(-1)).half()
