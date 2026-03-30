# Changelog

All notable changes to this project will be documented in this file.

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
