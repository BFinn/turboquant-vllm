"""Asymmetric decode attention using TurboQuant's inner product estimator.

During decode (one new query token per request), computes attention scores
directly from compressed KV data instead of reading the full FP16 cache:

    <q, k> ~ <q, k_mse> + ||r_k|| * sqrt(pi/2) / m * <S@q, sign(S@r_k)>

Term 1 is a standard Q @ K_mse^T matmul.
Term 2 is the QJL bias-correction: (Q @ S^T) @ signs^T * r_norm, scaled.

GQA: Qwen3.5 has 16 Q heads / 2 KV heads (8:1). The S@q projection is
computed once per KV head and reused across the 8 mapped Q heads.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .shadow_cache import ShadowKVCache


def turboquant_decode_attention(
    query: torch.Tensor,
    shadow_cache: ShadowKVCache,
    layer_idx: int,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    scale: float,
    output: torch.Tensor,
    num_actual_tokens: int,
) -> torch.Tensor:
    """Compute decode attention from TurboQuant shadow cache.

    Args:
        query:      (num_tokens_padded, num_q_heads, head_dim) - new decode queries
        shadow_cache: ShadowKVCache with compressed blocks
        layer_idx:  which model layer
        block_table: (batch_size, max_blocks) physical block indices
        seq_lens:   (batch_size,) sequence lengths including new token
        block_size: tokens per paged block
        scale:      attention scale (1/sqrt(head_dim))
        output:     (num_tokens_padded, num_q_heads * head_dim) pre-allocated output
        num_actual_tokens: number of real tokens (batch_size for decode)

    Returns:
        output tensor with attention results written in-place
    """
    config = shadow_cache.config
    num_kv_heads = config.num_kv_heads
    num_q_heads = config.num_q_heads
    heads_per_kv = config.heads_per_kv
    head_dim = config.head_dim
    batch_size = seq_lens.shape[0]

    # vLLM may pass query as (num_tokens, num_q_heads * head_dim) or
    # (num_tokens, num_q_heads, head_dim). Reshape to 3D if needed.
    if query.dim() == 2:
        query = query.view(query.shape[0], num_q_heads, head_dim)

    # Process each request in the batch
    for req_idx in range(batch_size):
        seq_len = seq_lens[req_idx].item()
        num_blocks = (seq_len + block_size - 1) // block_size
        req_block_indices = block_table[req_idx, :num_blocks].tolist()

        # Query for this request: (num_q_heads, head_dim)
        q = query[req_idx]  # (num_q_heads, head_dim)

        all_head_outputs = []

        for kv_h in range(num_kv_heads):
            # Gather compressed keys and decompressed values for this head
            k_mse, signs, r_norm = shadow_cache.gather_compressed_keys(
                layer_idx, req_block_indices, kv_h,
            )
            # Trim to actual seq_len (last block may be partially filled)
            k_mse = k_mse[:seq_len]    # (L, D) fp16
            signs = signs[:seq_len]     # (L, D) int8
            r_norm = r_norm[:seq_len]   # (L,) fp16

            values = shadow_cache.gather_decompressed_values(
                layer_idx, req_block_indices, kv_h,
            )
            values = values[:seq_len]   # (L, D) fp16

            # Get the S matrix for this KV head's compressor
            S = shadow_cache.key_compressors[layer_idx][kv_h].S
            correction_scale = shadow_cache.key_compressors[layer_idx][kv_h].correction_scale

            # Precompute k_mse and signs in float for GEMMs
            k_mse_f = k_mse.float()       # (L, D)
            signs_f = signs.float()         # (L, D)
            r_norm_f = r_norm.float()       # (L,)

            # For each Q head mapped to this KV head
            q_start = kv_h * heads_per_kv
            q_end = q_start + heads_per_kv

            # Batch all Q heads for this KV group: (heads_per_kv, D)
            q_group = q[q_start:q_end].float()

            # Term 1: Q @ K_mse^T -> (heads_per_kv, L)
            term1 = q_group @ k_mse_f.T

            # Term 2: QJL correction
            # Project queries through S: (heads_per_kv, D) @ (D, D) -> (heads_per_kv, D)
            q_proj = q_group @ S.T
            # Dot with signs: (heads_per_kv, D) @ (L, D)^T -> (heads_per_kv, L)
            qjl_ip = q_proj @ signs_f.T
            # Scale by residual norms: (heads_per_kv, L) * (L,) broadcast
            term2 = correction_scale * qjl_ip * r_norm_f.unsqueeze(0)

            # Combined scores -> softmax -> weighted sum
            scores = (term1 + term2) * scale  # (heads_per_kv, L)
            weights = F.softmax(scores, dim=-1)  # (heads_per_kv, L)

            # Weighted sum of values: (heads_per_kv, L) @ (L, D) -> (heads_per_kv, D)
            head_out = weights @ values.float()
            all_head_outputs.append(head_out)

        # Stack all Q heads: (num_q_heads, head_dim)
        req_output = torch.cat(all_head_outputs, dim=0)  # (num_q_heads, D)

        # Write to output: may be (num_tokens, num_q_heads * head_dim)
        # or (num_tokens, num_q_heads, head_dim)
        if output.dim() == 3:
            output[req_idx] = req_output.to(output.dtype)
        else:
            output[req_idx, :] = req_output.reshape(-1).to(output.dtype)

    return output
