"""Shadow KV cache: compressed storage indexed by (layer_idx, block_idx).

Mirrors vLLM's paged block structure but stores TurboQuant-compressed data.
Each CompressedBlock holds per-head compressed keys and values for one
paged block (block_size tokens).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

import torch

from .config import TurboQuantConfig
from .compressor import TQKeyCompressorGPU, TQValueCompressorGPU

logger = logging.getLogger(__name__)


@dataclass
class CompressedBlock:
    """Compressed KV data for one paged block, per KV head."""
    num_valid: int
    # Per-head lists (length = num_kv_heads)
    key_mse: list[torch.Tensor]       # [(num_valid, head_dim) fp16]
    key_qjl_signs: list[torch.Tensor] # [(num_valid, head_dim) int8]
    key_r_norm: list[torch.Tensor]    # [(num_valid,) fp16]
    val_indices: list[torch.Tensor]   # [(num_valid, head_dim) uint8]
    val_norms: list[torch.Tensor]     # [(num_valid,) fp16]


class ShadowKVCache:
    """Compressed KV cache that shadows vLLM's FP16 paged blocks.

    Stores one CompressedBlock per (layer_idx, block_idx) for full-attention
    layers only. Provides methods to compress blocks and gather compressed
    data across blocks for decode attention.
    """

    def __init__(self, config: TurboQuantConfig, device: torch.device):
        self.config = config
        self.device = device

        # Per-layer, per-head compressors
        self.key_compressors: dict[int, list[TQKeyCompressorGPU]] = {}
        self.val_compressors: dict[int, list[TQValueCompressorGPU]] = {}

        for layer_idx in config.full_attn_layers:
            k_comps = []
            v_comps = []
            for h in range(config.num_kv_heads):
                seed = config.seed_base + layer_idx * 1000 + h
                k_comps.append(TQKeyCompressorGPU(
                    config.head_dim, config.bits, seed, device,
                ))
                v_comps.append(TQValueCompressorGPU(
                    config.head_dim, config.bits, seed + 500, device,
                ))
            self.key_compressors[layer_idx] = k_comps
            self.val_compressors[layer_idx] = v_comps

        # Compressed block storage
        self.blocks: dict[tuple[int, int], CompressedBlock] = {}

    def compress_and_store(
        self,
        layer_idx: int,
        block_idx: int,
        key_block: torch.Tensor,
        val_block: torch.Tensor,
        num_valid: int,
    ) -> None:
        """Compress a KV cache block and store it.

        Args:
            layer_idx: model layer index (must be in full_attn_layers)
            block_idx: physical block index in vLLM's paged cache
            key_block: (block_size, num_kv_heads, head_dim) from paged cache
            val_block: (block_size, num_kv_heads, head_dim) from paged cache
            num_valid: number of actually filled token slots in this block
        """
        if num_valid == 0:
            return

        k_mse_list, k_signs_list, k_rnorm_list = [], [], []
        v_idx_list, v_norms_list = [], []

        for h in range(self.config.num_kv_heads):
            keys_h = key_block[:num_valid, h, :]  # (num_valid, D)
            vals_h = val_block[:num_valid, h, :]

            k_compressed = self.key_compressors[layer_idx][h].compress(keys_h)
            k_mse_list.append(k_compressed["k_mse"])
            k_signs_list.append(k_compressed["qjl_signs"])
            k_rnorm_list.append(k_compressed["r_norm"])

            v_compressed = self.val_compressors[layer_idx][h].compress(vals_h)
            v_idx_list.append(v_compressed["indices"])
            v_norms_list.append(v_compressed["norms"])

        self.blocks[(layer_idx, block_idx)] = CompressedBlock(
            num_valid=num_valid,
            key_mse=k_mse_list,
            key_qjl_signs=k_signs_list,
            key_r_norm=k_rnorm_list,
            val_indices=v_idx_list,
            val_norms=v_norms_list,
        )

    def gather_compressed_keys(
        self,
        layer_idx: int,
        block_indices: list[int],
        kv_head: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather compressed key data across blocks for one KV head.

        Returns:
            k_mse:      (total_tokens, head_dim) fp16
            qjl_signs:  (total_tokens, head_dim) int8
            r_norm:     (total_tokens,) fp16
        """
        k_mse_parts, signs_parts, rnorm_parts = [], [], []
        for blk_idx in block_indices:
            cb = self.blocks.get((layer_idx, blk_idx))
            if cb is None:
                continue
            k_mse_parts.append(cb.key_mse[kv_head])
            signs_parts.append(cb.key_qjl_signs[kv_head])
            rnorm_parts.append(cb.key_r_norm[kv_head])

        return (
            torch.cat(k_mse_parts, dim=0),
            torch.cat(signs_parts, dim=0),
            torch.cat(rnorm_parts, dim=0),
        )

    def gather_decompressed_values(
        self,
        layer_idx: int,
        block_indices: list[int],
        kv_head: int,
    ) -> torch.Tensor:
        """Gather and decompress values across blocks for one KV head.

        Returns: (total_tokens, head_dim) fp16
        """
        parts = []
        for blk_idx in block_indices:
            cb = self.blocks.get((layer_idx, blk_idx))
            if cb is None:
                continue
            v_decompressed = self.val_compressors[layer_idx][kv_head].decompress({
                "indices": cb.val_indices[kv_head],
                "norms": cb.val_norms[kv_head],
            })
            parts.append(v_decompressed)

        return torch.cat(parts, dim=0)

    def evict(self, layer_idx: int, block_idx: int) -> None:
        """Remove compressed data for a freed block."""
        self.blocks.pop((layer_idx, block_idx), None)

    def has_block(self, layer_idx: int, block_idx: int) -> bool:
        return (layer_idx, block_idx) in self.blocks
