# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1] - 2026-03-31

### Changed

- Batch value decompression: gather all blocks then single matmul instead of per-block decompress (+29% medium-context throughput)
- Batch KV heads in decode attention via `torch.bmm` instead of Python loop
- Use model-native dtype (bf16/fp16) throughout decode path instead of fp32 upcasts
- Store QJL signs as fp16 instead of int8 to avoid per-step cast overhead
- Incremental block compression: only compress newly written tokens, skip already-compressed slots

### Performance

Benchmarked on RTX 5080 16 GB, Qwen3.5-35B-A3B AWQ with MoE CPU offloading:

| Context | Before | After | Improvement |
|---------|--------|-------|-------------|
| Short (~13 tokens) | 12.7 tok/s | 13.8 tok/s | +8.7% |
| Medium (~500 tokens) | 10.1 tok/s | 12.7 tok/s | +25.7% |

## [0.1.0] - 2026-03-29

### Added

- Core TurboQuant implementation: Lloyd-Max solver, random rotation, QJL residual correction
- `TurboQuantMSE` (Stage 1) and `TurboQuantProd` (Stage 1+2) quantizers
- Production compressors for real model KV cache tensors
- vLLM plugin via `vllm.general_plugins` entry point (prefill: FP16, decode: TurboQuant)
- Environment-variable configuration for vLLM plugin
- Disk-cached codebook solver to avoid recomputation
- Pytest test suite with synthetic and algorithm validation tests
- CI workflow (GitHub Actions) with Python 3.10-3.12, ruff lint, pytest
- Validated against Qwen2.5-3B-Instruct KV cache (2K-8K context)
