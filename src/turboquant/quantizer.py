"""Core TurboQuant quantizers: MSE-optimal and inner-product-unbiased.

Stage 1 (TurboQuantMSE): Random rotation + per-coordinate Lloyd-Max quantization.
Stage 2 (TurboQuantProd): Adds 1-bit QJL residual correction for unbiased dot products.

Reference: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
           (ICLR 2026, https://arxiv.org/abs/2504.19874)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .lloyd_max import LloydMaxCodebook
from .rotation import generate_qjl_matrix, generate_rotation_matrix


class TurboQuantMSE(nn.Module):
    """Stage 1: MSE-optimal quantizer.

    Randomly rotates input vectors, then applies per-coordinate Lloyd-Max
    quantization. Reconstruction is the reverse: look up centroids, unrotate.
    """

    def __init__(self, d: int, bits: int, seed: int = 42, device: str = "cpu"):
        super().__init__()
        self.d = d
        self.bits = bits
        self.device = device
        self.register_buffer("Pi", generate_rotation_matrix(d, seed=seed, device=device))
        self.codebook = LloydMaxCodebook(d, bits)
        self.register_buffer("centroids", self.codebook.centroids.to(device))
        self.register_buffer("boundaries", self.codebook.boundaries.to(device))

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.Pi.T

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        return y @ self.Pi

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize vectors to codebook indices."""
        y = self.rotate(x)
        diffs = y.unsqueeze(-1) - self.centroids
        return diffs.abs().argmin(dim=-1)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Reconstruct vectors from codebook indices."""
        y_hat = self.centroids[indices]
        return self.unrotate(y_hat)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full quantize-dequantize cycle. Returns (reconstructed, indices)."""
        indices = self.quantize(x)
        x_hat = self.dequantize(indices)
        return x_hat, indices


class TurboQuantProd(nn.Module):
    """Stage 1 + Stage 2: Unbiased inner product quantizer.

    Uses (b-1)-bit MSE quantizer + 1-bit QJL on residuals.
    The combined estimator for <query, key> is:

        <q, k_mse> + ||residual|| * sqrt(pi/2) / m * <S @ q, sign(S @ residual)>

    This is mathematically unbiased with variance O(1/d).
    """

    def __init__(
        self,
        d: int,
        bits: int,
        qjl_dim: int | None = None,
        seed: int = 42,
        device: str = "cpu",
    ):
        super().__init__()
        self.d = d
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.qjl_dim = qjl_dim or d
        self.device = device
        self.mse = TurboQuantMSE(d, self.mse_bits, seed=seed, device=device)
        self.register_buffer(
            "S",
            generate_qjl_matrix(d, m=self.qjl_dim, seed=seed + 1, device=device),
        )

    def quantize(self, x: torch.Tensor) -> dict:
        """Full TurboQuant quantization.

        Returns:
            Dict with 'mse_indices', 'qjl_signs', and 'residual_norm'.
        """
        x_hat, mse_indices = self.mse(x)
        residual = x - x_hat
        residual_norm = torch.norm(residual, dim=-1, keepdim=True)
        projected = residual @ self.S.T
        qjl_signs = torch.sign(projected)
        qjl_signs[qjl_signs == 0] = 1.0
        return {
            "mse_indices": mse_indices,
            "qjl_signs": qjl_signs,
            "residual_norm": residual_norm.squeeze(-1),
        }

    def dequantize(self, compressed: dict) -> torch.Tensor:
        """Reconstruct vectors from MSE component (lossy)."""
        return self.mse.dequantize(compressed["mse_indices"])

    def inner_product(self, y: torch.Tensor, compressed: dict) -> torch.Tensor:
        """Compute unbiased inner product estimate <y, x> from compressed x.

        Uses the QJL-corrected asymmetric estimator from the paper.
        """
        x_mse = self.mse.dequantize(compressed["mse_indices"])
        term1 = (y * x_mse).sum(dim=-1)
        y_projected = y @ self.S.T
        qjl_ip = (y_projected * compressed["qjl_signs"]).sum(dim=-1)
        correction_scale = math.sqrt(math.pi / 2) / self.qjl_dim
        term2 = compressed["residual_norm"] * correction_scale * qjl_ip
        return term1 + term2

    def forward(self, x: torch.Tensor) -> dict:
        """Quantize input vectors."""
        return self.quantize(x)
