#!/usr/bin/env bash
# TurboQuant + Qwen3.5-35B-A3B vLLM server for OpenClaw agents
# Usage: ./serve.sh [port]  (default: 8082)
#
# Prerequisites:
#   pip install -e /home/bigfinn/scratch/turboquant-pytorch
#   vLLM 0.18.x with patched re-initialization assertion (see below)
#
# vLLM 0.18 patch required for hybrid Mamba+Attention + CPU offloading:
#   File: .venv/.../vllm/v1/worker/gpu_model_runner.py
#   Change the "Cannot re-initialize the input batch" assert to a warning.
#   See: https://github.com/vllm-project/vllm/pull/18298
#
# Memory budget (RTX 5080 16GB):
#   Model: ~2.5 GiB on GPU (20 GB offloaded to CPU RAM)
#   KV cache: ~13 GiB → 131K tokens capacity (TQ 3-bit compressed)
#   max_model_len: 131072
#
# Model config (Qwen3.5-35B-A3B = hybrid MoE, 256 experts, 8 active):
#   head_dim=256, num_kv_heads=2, num_q_heads=16, 40 layers
#   10 full-attention layers (TQ compressed): 3,7,11,15,19,23,27,31,35,39
#   30 linear-attention (GDN) layers: uncompressed

set -euo pipefail
PORT="${1:-8082}"

export VLLM_TURBOQUANT_ENABLED=1
export VLLM_TURBOQUANT_BITS=3
export VLLM_TURBOQUANT_HEAD_DIM=256
export VLLM_TURBOQUANT_NUM_KV_HEADS=2
export VLLM_TURBOQUANT_NUM_Q_HEADS=16

exec vllm serve cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit \
  --port "$PORT" \
  --enforce-eager \
  --gpu-memory-utilization 0.95 \
  --cpu-offload-gb 20 \
  --max-model-len 131072 \
  --max-num-seqs 2 \
  --trust-remote-code \
  --served-model-name Qwen3.5-35B-A3B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
