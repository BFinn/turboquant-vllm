"""TurboQuant plugin configuration parsed from environment variables."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Default Qwen3.5 35B A3B full-attention layers (every 4th layer, 0-indexed)
_DEFAULT_FULL_ATTN_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39)


@dataclass
class TurboQuantConfig:
    bits: int = 3
    enabled: bool = True
    full_attn_layers: tuple[int, ...] = _DEFAULT_FULL_ATTN_LAYERS
    head_dim: int = 128
    num_kv_heads: int = 2
    num_q_heads: int = 16
    seed_base: int = 42

    @classmethod
    def from_env(cls) -> TurboQuantConfig:
        enabled = os.environ.get("VLLM_TURBOQUANT_ENABLED", "1") != "0"
        bits = int(os.environ.get("VLLM_TURBOQUANT_BITS", "3"))
        head_dim = int(os.environ.get("VLLM_TURBOQUANT_HEAD_DIM", "128"))
        num_kv_heads = int(os.environ.get("VLLM_TURBOQUANT_NUM_KV_HEADS", "2"))
        num_q_heads = int(os.environ.get("VLLM_TURBOQUANT_NUM_Q_HEADS", "16"))
        seed_base = int(os.environ.get("VLLM_TURBOQUANT_SEED", "42"))

        layers_str = os.environ.get("VLLM_TURBOQUANT_LAYERS", "")
        if layers_str:
            full_attn_layers = tuple(int(x) for x in layers_str.split(","))
        else:
            full_attn_layers = _DEFAULT_FULL_ATTN_LAYERS

        if bits < 2 or bits > 4:
            raise ValueError(f"VLLM_TURBOQUANT_BITS must be 2-4, got {bits}")

        if "VLLM_TURBOQUANT_HEAD_DIM" not in os.environ:
            logger.warning(
                "VLLM_TURBOQUANT_HEAD_DIM not set, defaulting to 128. "
                "Set this to your model's attention head dimension "
                "(e.g. 128 for Llama/Qwen/Mistral, 256 for Qwen3.5-35B)."
            )

        return cls(
            bits=bits,
            enabled=enabled,
            full_attn_layers=full_attn_layers,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            num_q_heads=num_q_heads,
            seed_base=seed_base,
        )

    @property
    def mse_bits(self) -> int:
        """Bits allocated to MSE stage (total - 1 for QJL)."""
        return max(self.bits - 1, 1)

    @property
    def heads_per_kv(self) -> int:
        """Number of Q heads per KV head (GQA ratio)."""
        return self.num_q_heads // self.num_kv_heads
