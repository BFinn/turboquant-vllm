"""KV cache wrapper using TurboQuant compression.

Uses TurboQuantProd for keys (need unbiased inner products for attention)
and TurboQuantMSE for values (need MSE reconstruction for weighted sums).
"""

from __future__ import annotations

import torch

from .quantizer import TurboQuantMSE, TurboQuantProd


class TurboQuantKVCache:
    """Drop-in replacement concept for a standard KV cache with TurboQuant compression."""

    def __init__(
        self,
        d_key: int,
        d_value: int,
        bits: int = 3,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.d_key = d_key
        self.d_value = d_value
        self.bits = bits
        self.device = device
        self.key_quantizer = TurboQuantProd(d_key, bits, seed=seed, device=device)
        self.value_quantizer = TurboQuantMSE(d_value, bits, seed=seed + 100, device=device)
        self.key_cache: list[dict] = []
        self.value_cache: list[dict] = []

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Append key-value pairs. Shapes: (seq_len, d) or (batch, seq_len, d)."""
        orig_shape = keys.shape
        flat_keys = keys.reshape(-1, self.d_key)
        flat_values = values.reshape(-1, self.d_value)
        compressed_keys = self.key_quantizer.quantize(flat_keys)
        value_indices = self.value_quantizer.quantize(flat_values)
        self.key_cache.append(
            {
                "mse_indices": compressed_keys["mse_indices"],
                "qjl_signs": compressed_keys["qjl_signs"],
                "residual_norm": compressed_keys["residual_norm"],
                "shape": orig_shape,
            }
        )
        self.value_cache.append({"indices": value_indices, "shape": values.shape})

    def attention_scores(self, queries: torch.Tensor) -> torch.Tensor:
        """Compute attention scores using unbiased inner product estimation."""
        scores = []
        for cached in self.key_cache:
            s = self.key_quantizer.inner_product(queries, cached)
            scores.append(s)
        return torch.cat(scores, dim=-1) if scores else torch.tensor([])

    def get_values(self) -> torch.Tensor:
        """Reconstruct all cached values."""
        values = []
        for cached in self.value_cache:
            v = self.value_quantizer.dequantize(cached["indices"])
            values.append(v)
        return torch.cat(values, dim=0) if values else torch.tensor([])

    def memory_usage_bits(self) -> dict:
        """Estimate memory usage in bits."""
        n_keys = sum(c["mse_indices"].numel() for c in self.key_cache) if self.key_cache else 0
        n_qjl = sum(c["qjl_signs"].numel() for c in self.key_cache) if self.key_cache else 0
        n_norms = sum(c["residual_norm"].numel() for c in self.key_cache) if self.key_cache else 0
        n_values = sum(c["indices"].numel() for c in self.value_cache) if self.value_cache else 0
        key_bits = n_keys * self.key_quantizer.mse_bits + n_qjl * 1 + n_norms * 16
        value_bits = n_values * self.bits
        fp16_equivalent = (n_keys + n_values) * 16
        total = key_bits + value_bits
        return {
            "key_bits": key_bits,
            "value_bits": value_bits,
            "total_bits": total,
            "fp16_bits": fp16_equivalent,
            "compression_ratio": fp16_equivalent / total if total > 0 else 0,
        }

    def __len__(self) -> int:
        return sum(c["mse_indices"].shape[0] for c in self.key_cache) if self.key_cache else 0
