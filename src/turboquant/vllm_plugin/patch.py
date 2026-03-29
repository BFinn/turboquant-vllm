"""Monkey-patch for FlashAttentionImpl to intercept KV cache operations.

Patches two methods:
- do_kv_cache_update: After FP16 write, also compress into shadow cache
- forward: During decode on TQ layers, use asymmetric attention from shadow cache

DeltaNet layers and prefill are completely untouched.
"""

from __future__ import annotations

import logging
import re

import torch

from .config import TurboQuantConfig
from .decode_attention import turboquant_decode_attention
from .shadow_cache import ShadowKVCache

logger = logging.getLogger(__name__)

# Module-level state (set by apply_patch)
_shadow_cache: ShadowKVCache | None = None
_config: TurboQuantConfig | None = None
_original_forward = None
_original_do_kv_cache_update = None

# Regex to extract layer index from names like "model.layers.3.self_attn"
_LAYER_IDX_RE = re.compile(r"\.layers\.(\d+)\.")


def _extract_layer_idx(layer: torch.nn.Module) -> int | None:
    """Extract numeric layer index from the Attention module's layer_name."""
    name = getattr(layer, "layer_name", None)
    if name is None:
        return None
    m = _LAYER_IDX_RE.search(name)
    return int(m.group(1)) if m else None


def _patched_do_kv_cache_update(
    self,
    layer: torch.nn.Module,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """After writing FP16 to paged blocks, compress into shadow cache."""
    # Always do the original FP16 write (needed for prefill path)
    _original_do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping)

    layer_idx = _extract_layer_idx(layer)
    if layer_idx is None or layer_idx not in _config.full_attn_layers:
        return

    # Skip during CUDA graph capture — GPU-to-CPU sync is not allowed
    if torch.cuda.is_current_stream_capturing():
        return

    # Determine which physical blocks were affected
    # kv_cache shape: [2, num_blocks, block_size, num_kv_heads, head_dim]
    block_size = kv_cache.shape[2]
    key_cache = kv_cache[0]  # [num_blocks, block_size, num_kv_heads, head_dim]

    # slot_mapping: (num_actual_tokens,) maps each token to a slot index
    # block_idx = slot // block_size
    block_indices = (slot_mapping // block_size).unique().tolist()

    for block_idx in block_indices:
        if block_idx < 0:
            continue  # padding slots are -1
        k_block = key_cache[block_idx]  # (block_size, num_kv_heads, head_dim)
        v_block = kv_cache[1, block_idx]  # (block_size, num_kv_heads, head_dim)

        # Count valid tokens: slots in this block that appear in slot_mapping
        block_slots = slot_mapping[
            (slot_mapping >= block_idx * block_size) & (slot_mapping < (block_idx + 1) * block_size)
        ]
        if block_slots.numel() == 0:
            continue

        # The highest slot offset in this block tells us how many tokens are valid
        max_offset = (block_slots % block_size).max().item() + 1
        # But there may be earlier tokens already in the block from prior steps
        existing = _shadow_cache.blocks.get((layer_idx, block_idx))
        num_valid = max(max_offset, existing.num_valid if existing else 0)

        _shadow_cache.compress_and_store(
            layer_idx,
            block_idx,
            k_block,
            v_block,
            num_valid,
        )


def _patched_forward(
    self,
    layer: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    output: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Route to asymmetric attention for decode on TQ layers."""
    layer_idx = _extract_layer_idx(layer)

    # Not a TQ layer -> original path
    if layer_idx is None or layer_idx not in _config.full_attn_layers:
        return _original_forward(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )

    # Warmup / dummy run / CUDA graph capture -> fall back to original
    if attn_metadata is None or torch.cuda.is_current_stream_capturing():
        return _original_forward(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )

    # Prefill (max_query_len > 1) -> original FlashAttention
    # The FP16 paged cache was already written by do_kv_cache_update
    if attn_metadata.max_query_len > 1:
        return _original_forward(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )

    # === DECODE PATH: asymmetric attention from shadow cache ===
    assert output is not None, "Output tensor must be provided."

    block_size = kv_cache.shape[2]
    num_actual_tokens = attn_metadata.num_actual_tokens

    return turboquant_decode_attention(
        query=query,
        shadow_cache=_shadow_cache,
        layer_idx=layer_idx,
        block_table=attn_metadata.block_table,
        seq_lens=attn_metadata.seq_lens,
        block_size=block_size,
        scale=self.scale,
        output=output,
        num_actual_tokens=num_actual_tokens,
    )


def apply_patch(config: TurboQuantConfig) -> None:
    """Install the monkey-patch on FlashAttentionImpl. Re-entrant safe."""
    global _shadow_cache, _config, _original_forward, _original_do_kv_cache_update

    if _original_forward is not None:
        logger.debug("TurboQuant patch already applied, skipping")
        return

    _config = config
    _shadow_cache = ShadowKVCache(config, device=torch.device("cuda"))

    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    _original_forward = FlashAttentionImpl.forward
    _original_do_kv_cache_update = FlashAttentionImpl.do_kv_cache_update

    FlashAttentionImpl.forward = _patched_forward
    FlashAttentionImpl.do_kv_cache_update = _patched_do_kv_cache_update

    logger.info(
        "TurboQuant patched FlashAttentionImpl (layers=%s, bits=%d)",
        config.full_attn_layers,
        config.bits,
    )
