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
    dtype = query.dtype  # match model dtype (bf16 or fp16)

    # vLLM may pass query as (num_tokens, num_q_heads * head_dim) or
    # (num_tokens, num_q_heads, head_dim). Reshape to 3D if needed.
    if query.dim() == 2:
        query = query.view(query.shape[0], num_q_heads, head_dim)

    # Process each request in the batch
    for req_idx in range(batch_size):
        seq_len = seq_lens[req_idx].item()
        num_blocks = (seq_len + block_size - 1) // block_size
        req_block_indices = block_table[req_idx, :num_blocks].tolist()

        # Query for this request: (num_q_heads, head_dim) in model dtype
        q = query[req_idx]

        # Gather all KV heads and batch into (num_kv_heads, L, D)
        k_mse_all, signs_all, r_norm_all, values_all = [], [], [], []
        S_all, corr_scales = [], []
        for kv_h in range(num_kv_heads):
            key_indices, key_norms, signs, r_norm = (
                shadow_cache.gather_compressed_keys(
                    layer_idx, req_block_indices, kv_h))
            # Reconstruct k_mse on-the-fly from compact indices
            compressor = shadow_cache.key_compressors[layer_idx][kv_h]
            k_mse = compressor.reconstruct_k_mse(
                key_indices[:seq_len], key_norms[:seq_len], dtype)
            k_mse_all.append(k_mse)
            signs_all.append(signs[:seq_len].to(dtype))
            r_norm_all.append(r_norm[:seq_len].to(dtype))

            values = shadow_cache.gather_decompressed_values(
                layer_idx, req_block_indices, kv_h)
            values_all.append(values[:seq_len].to(dtype))

            S_all.append(compressor.S.to(dtype))
            corr_scales.append(compressor.correction_scale)

        # Stack: (num_kv_heads, L, D)
        k_mse_batch = torch.stack(k_mse_all)
        signs_batch = torch.stack(signs_all)
        r_norm_batch = torch.stack(r_norm_all)
        values_batch = torch.stack(values_all)
        S_batch = torch.stack(S_all)

        # Reshape query for GQA: (num_kv_heads, heads_per_kv, D)
        q_grouped = q.view(num_kv_heads, heads_per_kv, head_dim)

        # Term 1: batched Q @ K_mse^T -> (H_kv, heads_per_kv, L)
        term1 = torch.bmm(q_grouped, k_mse_batch.transpose(1, 2))

        # Term 2: QJL correction via batched matmuls
        q_proj = torch.bmm(q_grouped, S_batch.transpose(1, 2))
        qjl_ip = torch.bmm(q_proj, signs_batch.transpose(1, 2))
        # Scale by per-head correction_scale * residual norms
        for kv_h in range(num_kv_heads):
            qjl_ip[kv_h] *= corr_scales[kv_h] * r_norm_batch[kv_h].unsqueeze(0)

        # Combined scores -> softmax (fp32 for stability) -> weighted sum
        scores = (term1 + qjl_ip) * scale
        weights = F.softmax(scores.float(), dim=-1).to(dtype)

        # Weighted sum: (H_kv, heads_per_kv, L) @ (H_kv, L, D) -> (H_kv, heads_per_kv, D)
        head_out = torch.bmm(weights, values_batch)

        # Flatten to (num_q_heads, D)
        req_output = head_out.view(num_q_heads, head_dim)

        # Write to output
        if output.dim() == 3:
            output[req_idx] = req_output.to(output.dtype)
        else:
            output[req_idx, :] = req_output.reshape(-1).to(output.dtype)

    return output
