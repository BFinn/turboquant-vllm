"""Shadow KV cache: compressed storage indexed by (layer_idx, block_idx).

Mirrors vLLM's paged block structure but stores TurboQuant-compressed data.
Each CompressedBlock holds per-head compressed keys and values for one
paged block (block_size tokens).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from .compressor import TQKeyCompressorGPU, TQValueCompressorGPU
from .config import TurboQuantConfig

logger = logging.getLogger(__name__)


@dataclass
class CompressedBlock:
    """Compressed KV data for one paged block, per KV head."""

    num_valid: int
    # Per-head lists (length = num_kv_heads)
    key_indices: list[torch.Tensor]  # [(num_valid, head_dim) uint8]
    key_norms: list[torch.Tensor]  # [(num_valid,) fp16]
    key_qjl_signs: list[torch.Tensor]  # [(num_valid, head_dim) int8]
    key_r_norm: list[torch.Tensor]  # [(num_valid,) fp16]
    val_indices: list[torch.Tensor]  # [(num_valid, head_dim) uint8]
    val_norms: list[torch.Tensor]  # [(num_valid,) fp16]


class ShadowKVCache:
    """Compressed KV cache that shadows vLLM's FP16 paged blocks.

    Stores one CompressedBlock per (layer_idx, block_idx) for full-attention
    layers only. Provides methods to compress blocks and gather compressed
    data across blocks for decode attention.

    Compressors are created lazily on first use to avoid allocating GPU
    memory in the API server parent process (which never compresses).
    """

    def __init__(self, config: TurboQuantConfig, device: torch.device):
        self.config = config
        self.device = device

        # Per-layer, per-head compressors (created lazily)
        self.key_compressors: dict[int, list[TQKeyCompressorGPU]] = {}
        self.val_compressors: dict[int, list[TQValueCompressorGPU]] = {}
        self._compressors_initialized = False

        # Compressed block storage
        self.blocks: dict[tuple[int, int], CompressedBlock] = {}

    def _ensure_compressors(self) -> None:
        """Lazily create compressors on first use (in the EngineCore process)."""
        if self._compressors_initialized:
            return
        self._compressors_initialized = True

        for layer_idx in self.config.full_attn_layers:
            k_comps = []
            v_comps = []
            for h in range(self.config.num_kv_heads):
                seed = self.config.seed_base + layer_idx * 1000 + h
                k_comps.append(
                    TQKeyCompressorGPU(
                        self.config.head_dim,
                        self.config.bits,
                        seed,
                        self.device,
                    )
                )
                v_comps.append(
                    TQValueCompressorGPU(
                        self.config.head_dim,
                        self.config.bits,
                        seed + 500,
                        self.device,
                    )
                )
            self.key_compressors[layer_idx] = k_comps
            self.val_compressors[layer_idx] = v_comps

        logger.info("TurboQuant compressors initialized on %s", self.device)

    def compress_and_store(
        self,
        layer_idx: int,
        block_idx: int,
        key_block: torch.Tensor,
        val_block: torch.Tensor,
        num_valid: int,
    ) -> None:
        """Compress a KV cache block and store it (incrementally).

        Only compresses tokens that haven't been compressed yet.

        Args:
            layer_idx: model layer index (must be in full_attn_layers)
            block_idx: physical block index in vLLM's paged cache
            key_block: (block_size, num_kv_heads, head_dim) from paged cache
            val_block: (block_size, num_kv_heads, head_dim) from paged cache
            num_valid: number of actually filled token slots in this block
        """
        if num_valid == 0:
            return

        self._ensure_compressors()

        existing = self.blocks.get((layer_idx, block_idx))
        old_valid = existing.num_valid if existing else 0

        if num_valid <= old_valid:
            return  # nothing new to compress

        k_idx_list, k_norms_list, k_signs_list, k_rnorm_list = [], [], [], []
        v_idx_list, v_norms_list = [], []

        for h in range(self.config.num_kv_heads):
            keys_h = key_block[old_valid:num_valid, h, :]
            vals_h = val_block[old_valid:num_valid, h, :]

            k_compressed = self.key_compressors[layer_idx][h].compress(keys_h)
            v_compressed = self.val_compressors[layer_idx][h].compress(vals_h)

            if existing:
                k_idx_list.append(torch.cat([existing.key_indices[h], k_compressed["key_indices"]], dim=0))
                k_norms_list.append(torch.cat([existing.key_norms[h], k_compressed["key_norms"]], dim=0))
                k_signs_list.append(torch.cat([existing.key_qjl_signs[h], k_compressed["qjl_signs"]], dim=0))
                k_rnorm_list.append(torch.cat([existing.key_r_norm[h], k_compressed["r_norm"]], dim=0))
                v_idx_list.append(torch.cat([existing.val_indices[h], v_compressed["indices"]], dim=0))
                v_norms_list.append(torch.cat([existing.val_norms[h], v_compressed["norms"]], dim=0))
            else:
                k_idx_list.append(k_compressed["key_indices"])
                k_norms_list.append(k_compressed["key_norms"])
                k_signs_list.append(k_compressed["qjl_signs"])
                k_rnorm_list.append(k_compressed["r_norm"])
                v_idx_list.append(v_compressed["indices"])
                v_norms_list.append(v_compressed["norms"])

        self.blocks[(layer_idx, block_idx)] = CompressedBlock(
            num_valid=num_valid,
            key_indices=k_idx_list,
            key_norms=k_norms_list,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather compressed key data across blocks for one KV head.

        Returns:
            key_indices: (total_tokens, head_dim) uint8
            key_norms:   (total_tokens,) fp16
            qjl_signs:   (total_tokens, head_dim) int8
            r_norm:      (total_tokens,) fp16
        """
        self._ensure_compressors()
        idx_parts, norm_parts, signs_parts, rnorm_parts = [], [], [], []
        for blk_idx in block_indices:
            cb = self.blocks.get((layer_idx, blk_idx))
            if cb is None:
                continue
            idx_parts.append(cb.key_indices[kv_head])
            norm_parts.append(cb.key_norms[kv_head])
            signs_parts.append(cb.key_qjl_signs[kv_head])
            rnorm_parts.append(cb.key_r_norm[kv_head])

        return (
            torch.cat(idx_parts, dim=0),
            torch.cat(norm_parts, dim=0),
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

        Batches all blocks into a single decompress call.

        Returns: (total_tokens, head_dim) fp16
        """
        self._ensure_compressors()
        idx_parts, norm_parts = [], []
        for blk_idx in block_indices:
            cb = self.blocks.get((layer_idx, blk_idx))
            if cb is None:
                continue
            idx_parts.append(cb.val_indices[kv_head])
            norm_parts.append(cb.val_norms[kv_head])

        all_indices = torch.cat(idx_parts, dim=0)
        all_norms = torch.cat(norm_parts, dim=0)
        return self.val_compressors[layer_idx][kv_head].decompress(
            {"indices": all_indices, "norms": all_norms}
        )

    def compress_token_direct(
        self,
        layer_idx: int,
        block_idx: int,
        slot_offset: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Compress a single decode token directly (no paged cache needed).

        Args:
            layer_idx: model layer index
            block_idx: physical block index
            slot_offset: token position within the block (0..block_size-1)
            key:   (num_kv_heads, head_dim) single token key
            value: (num_kv_heads, head_dim) single token value
        """
        self._ensure_compressors()

        existing = self.blocks.get((layer_idx, block_idx))
        old_valid = existing.num_valid if existing else 0
        num_valid = max(slot_offset + 1, old_valid)

        # Token already compressed in this block — nothing to append
        if slot_offset < old_valid:
            return

        # Safety: slot_offset must be contiguous (no gaps)
        if slot_offset > old_valid:
            logger.warning(
                "compress_token_direct: gap at layer=%d block=%d "
                "slot_offset=%d old_valid=%d — skipping",
                layer_idx, block_idx, slot_offset, old_valid,
            )
            return

        k_idx_list, k_norms_list, k_signs_list, k_rnorm_list = [], [], [], []
        v_idx_list, v_norms_list = [], []

        for h in range(self.config.num_kv_heads):
            k_compressed = self.key_compressors[layer_idx][h].compress(
                key[h].unsqueeze(0))
            v_compressed = self.val_compressors[layer_idx][h].compress(
                value[h].unsqueeze(0))

            if existing:
                k_idx_list.append(torch.cat([existing.key_indices[h], k_compressed["key_indices"]], dim=0))
                k_norms_list.append(torch.cat([existing.key_norms[h], k_compressed["key_norms"]], dim=0))
                k_signs_list.append(torch.cat([existing.key_qjl_signs[h], k_compressed["qjl_signs"]], dim=0))
                k_rnorm_list.append(torch.cat([existing.key_r_norm[h], k_compressed["r_norm"]], dim=0))
                v_idx_list.append(torch.cat([existing.val_indices[h], v_compressed["indices"]], dim=0))
                v_norms_list.append(torch.cat([existing.val_norms[h], v_compressed["norms"]], dim=0))
            else:
                k_idx_list.append(k_compressed["key_indices"])
                k_norms_list.append(k_compressed["key_norms"])
                k_signs_list.append(k_compressed["qjl_signs"])
                k_rnorm_list.append(k_compressed["r_norm"])
                v_idx_list.append(v_compressed["indices"])
                v_norms_list.append(v_compressed["norms"])

        self.blocks[(layer_idx, block_idx)] = CompressedBlock(
            num_valid=num_valid,
            key_indices=k_idx_list,
            key_norms=k_norms_list,
            key_qjl_signs=k_signs_list,
            key_r_norm=k_rnorm_list,
            val_indices=v_idx_list,
            val_norms=v_norms_list,
        )

    def evict(self, layer_idx: int, block_idx: int) -> None:
        """Remove compressed data for a freed block."""
        self.blocks.pop((layer_idx, block_idx), None)

    def has_block(self, layer_idx: int, block_idx: int) -> bool:
        return (layer_idx, block_idx) in self.blocks
