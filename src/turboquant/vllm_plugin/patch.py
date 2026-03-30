"""Monkey-patch for FlashAttentionImpl to intercept KV cache operations.

Patches `forward` to:
- After FP16 cache write (reshape_and_cache_flash), compress into shadow cache
- During decode on TQ layers, use asymmetric attention from shadow cache
  instead of flash_attn_varlen_func

DeltaNet layers and prefill are completely untouched.

Compatible with vLLM 0.8.x where FlashAttentionImpl.forward handles both
cache writes and attention computation in a single method.
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

# Regex to extract layer index from names like "model.layers.3.self_attn"
_LAYER_IDX_RE = re.compile(r"\.layers\.(\d+)\.")


def _extract_layer_idx(layer: torch.nn.Module) -> int | None:
    """Extract numeric layer index from the Attention module's layer_name."""
    name = getattr(layer, "layer_name", None)
    if name is None:
        return None
    m = _LAYER_IDX_RE.search(name)
    return int(m.group(1)) if m else None


def _compress_blocks(
    layer_idx: int,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Compress affected blocks into the shadow cache after FP16 cache write."""
    # kv_cache shape: [2, num_blocks, block_size, num_kv_heads, head_dim]
    block_size = kv_cache.shape[2]
    key_cache = kv_cache[0]  # [num_blocks, block_size, num_kv_heads, head_dim]

    block_indices = (slot_mapping // block_size).unique().tolist()

    for block_idx in block_indices:
        if block_idx < 0:
            continue  # padding slots are -1
        k_block = key_cache[block_idx]  # (block_size, num_kv_heads, head_dim)
        v_block = kv_cache[1, block_idx]  # (block_size, num_kv_heads, head_dim)

        # Count valid tokens in this block
        block_slots = slot_mapping[
            (slot_mapping >= block_idx * block_size)
            & (slot_mapping < (block_idx + 1) * block_size)
        ]
        if block_slots.numel() == 0:
            continue

        max_offset = (block_slots % block_size).max().item() + 1
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
    """Patched forward: compress KV on TQ layers, use asymmetric decode."""
    assert output is not None, "Output tensor must be provided."

    if attn_metadata is None:
        # Profiling run
        return output

    layer_idx = _extract_layer_idx(layer)
    is_tq_layer = layer_idx is not None and layer_idx in _config.full_attn_layers
    is_decode = attn_metadata.max_query_len == 1
    capturing = torch.cuda.is_current_stream_capturing()

    # --- Non-TQ layer: pass through entirely ---
    if not is_tq_layer:
        return _original_forward(
            self, layer, query, key, value, kv_cache, attn_metadata, output,
            output_scale, output_block_scale,
        )

    # --- TQ layer, prefill or CUDA graph capture: original path + compress ---
    if not is_decode or capturing:
        result = _original_forward(
            self, layer, query, key, value, kv_cache, attn_metadata, output,
            output_scale, output_block_scale,
        )
        if not capturing:
            _compress_blocks(layer_idx, kv_cache, attn_metadata.slot_mapping)
        return result

    # --- TQ layer, decode: write cache, compress, then asymmetric attention ---
    num_actual_tokens = attn_metadata.num_actual_tokens

    # Step 1: Write KV to paged cache (same as original)
    key_cache, value_cache = kv_cache.unbind(0)
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        attn_metadata.slot_mapping,
        self.kv_cache_dtype,
        layer._k_scale,
        layer._v_scale,
    )

    # Step 2: Compress the updated blocks
    _compress_blocks(layer_idx, kv_cache, attn_metadata.slot_mapping)

    # Step 3: Asymmetric attention from shadow cache
    block_size = kv_cache.shape[2]
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
    global _shadow_cache, _config, _original_forward

    if _original_forward is not None:
        logger.debug("TurboQuant patch already applied, skipping")
        return

    _config = config
    _shadow_cache = ShadowKVCache(config, device=torch.device("cuda"))

    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    _original_forward = FlashAttentionImpl.forward
    FlashAttentionImpl.forward = _patched_forward

    logger.info(
        "TurboQuant patched FlashAttentionImpl (layers=%s, bits=%d)",
        config.full_attn_layers,
        config.bits,
    )
