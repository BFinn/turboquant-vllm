# TurboQuant

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)

A PyTorch implementation of [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026), Google's two-stage vector quantization algorithm for compressing LLM key-value caches to 2-4 bits per coordinate.

## Key Features

- **3-7x KV cache compression** with minimal impact on attention accuracy (99.5%+ cosine similarity at 3-bit)
- **Two-stage quantization**: Lloyd-Max optimal scalar quantization with QJL residual correction for unbiased inner products
- **vLLM plugin**: drop-in KV cache compression via `vllm.general_plugins` entry point -- prefill uses standard FP16 attention, decode uses TurboQuant asymmetric attention
- **Validated on real models**: tested against Qwen2.5-3B-Instruct; production use with Qwen3.5-35B-A3B-AWQ at 131K context on RTX 5080 16 GB
- **No custom CUDA kernels**: pure PyTorch implementation

## Installation

From source (PyPI package coming soon):

```bash
pip install git+https://github.com/BFinn/turboquant-vllm.git
```

With vLLM plugin support:

```bash
pip install "turboquant[vllm] @ git+https://github.com/BFinn/turboquant-vllm.git"
```

Development (includes pytest and ruff):

```bash
git clone https://github.com/BFinn/turboquant-vllm.git
cd turboquant-vllm
pip install -e ".[dev]"
```

For CUDA-enabled PyTorch (if not already installed):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Quick Start

```python
import torch
from turboquant import TurboQuantProd, solve_lloyd_max

# Solve Lloyd-Max codebook for 3-bit quantization
# dim = head dimension (128 for Llama/Qwen2.5, 256 for Qwen3.5), bits = quantization bit-width
# Returns a 1D tensor of 2^bits optimal centroids for the post-rotation coordinate distribution
codebook = solve_lloyd_max(dim=128, bits=3)

# Create the quantizer (Stage 1 + Stage 2)
tq = TurboQuantProd(dim=128, bits=3, codebook=codebook)

# Quantize a batch of key vectors
keys = torch.randn(64, 128)  # 64 tokens, head_dim=128
compressed = tq.quantize(keys)

# Estimate inner products directly from compressed data (no decompression needed)
query = torch.randn(128)
scores = tq.inner_product(query, compressed)  # shape: (64,)
```

## Supported Models & Hardware

### Tested Models

| Model | head_dim | KV Heads | Layers (full-attn / total) | Status |
|-------|----------|----------|---------------------------|--------|
| Qwen3.5-35B-A3B-AWQ | 256 | 2 | 10 / 40 | Production (131K context on RTX 5080 16 GB) |
| Qwen2.5-3B-Instruct | 128 | 2 | 36 / 36 | Validated (benchmarks above) |

TurboQuant works with any model using standard multi-head or grouped-query attention. Set `VLLM_TURBOQUANT_HEAD_DIM` to match your model (128 for Llama/Qwen2.5/Mistral, 256 for Qwen3.5). Codebooks are pre-computed for both d=128 and d=256.

### Hardware

- **Required**: NVIDIA GPU with CUDA (tested on RTX 5080 16 GB)
- **Not supported**: CPU-only, AMD ROCm, Apple MPS
- **VRAM**: the plugin itself adds negligible overhead; your GPU just needs enough memory to run the model. The whole point is that compressed KV cache lets you fit longer contexts or larger batches in the same VRAM.

## How TurboQuant Differs from FP8 / INT4 KV Cache

vLLM supports `--kv-cache-dtype fp8` natively. Here's how TurboQuant compares:

| | FP8 KV Cache | INT4 (KIVI-style) | TurboQuant 3-bit |
|---|---|---|---|
| Compression | 2x (16→8 bits) | 4x (16→4 bits) | 5x (16→3 bits + 1-bit QJL) |
| Approach | Direct cast | Per-channel quantization | Rotation + Lloyd-Max + QJL correction |
| Inner product bias | None | Small | **Zero** (mathematically unbiased) |
| Attention fidelity | ~1.000 cosine sim | ~0.995 | 0.995 (3-bit) |
| Custom kernels needed | No (hardware FP8) | Yes | No (pure PyTorch) |
| Works with GQA | Yes | Yes | Yes |

**When to use TurboQuant over FP8**: when you need more than 2x compression. FP8 is simpler and lossless for most purposes, but it only halves KV memory. TurboQuant at 3-bit gives 5x compression with near-identical attention scores, which matters for long-context inference where KV cache is the memory bottleneck.

**When to stick with FP8**: short contexts where KV cache isn't the bottleneck, or when you need zero approximation error.

## Algorithm

### Stage 1: Random Rotation + Lloyd-Max Quantization

Each vector is multiplied by a random orthogonal matrix (generated via QR decomposition of a Gaussian matrix). This rotation is the key insight: it makes every coordinate of the resulting vector follow a predictable bell-curve distribution (Beta distribution, well-approximated by Gaussian N(0, 1/d) for typical head dimensions).

Because the distribution is known and coordinates become nearly independent, we can design an **optimal scalar quantizer** (Lloyd-Max) for each coordinate independently. The Lloyd-Max algorithm finds the best set of "buckets" to round values into, minimizing mean squared error. These codebooks are precomputed once per bit-width.

To quantize: rotate the vector, round each coordinate to its nearest codebook centroid, store the indices.
To dequantize: look up centroids, reverse the rotation.

### Stage 2: QJL Residual Correction (1 bit)

The MSE-optimal quantizer from Stage 1 introduces a small bias in dot products (inner products). Since attention scores are just dot products between queries and keys, this bias accumulates.

The Quantized Johnson-Lindenstrauss (QJL) transform fixes this. It takes the quantization residual (the error left over from Stage 1), projects it through a random Gaussian matrix, and stores just the **sign** (+1 or -1) of each projection -- exactly 1 bit per dimension. This single bit is enough to make the inner product estimate **mathematically unbiased**.

The combined estimator for `<query, key>` is:

```
<q, k> ~ <q, k_mse> + ||residual|| * sqrt(pi/2) / m * <S @ q, sign(S @ residual)>
```

Where `S` is the random projection matrix, `k_mse` is the Stage 1 reconstruction, and `residual = k - k_mse`.

### Why This Works Despite High Per-Vector Error

An important subtlety: the per-vector reconstruction error is significant (23-44% relative error depending on bit-width). If you decompress the vectors and feed them to standard attention, the model produces garbage.

But TurboQuant does not need accurate vector reconstruction. It needs accurate **inner products** (attention scores). The QJL correction ensures these are unbiased with variance O(1/d), where d is the head dimension (typically 128). The attention distribution over tokens is preserved even when individual vectors look quite different from the originals.

## Benchmark Results

### Synthetic Vector Tests

Core algorithm validation on random unit vectors (d=128, no model required).

**MSE Distortion** (1000 random unit vectors):

| Bits | Measured MSE | Paper Upper Bound | Ratio |
|------|-------------|-------------------|-------|
| 1    | 0.362       | 0.680             | 0.53x |
| 2    | 0.116       | 0.170             | 0.68x |
| 3    | 0.034       | 0.043             | 0.81x |
| 4    | 0.009       | 0.011             | 0.87x |

All measurements are well within the theoretical bounds. The ratio approaching 1.0 at higher bit-widths is expected -- the bound is tighter when quantization is finer.

**Inner Product Accuracy** (2000 random vector pairs):

| Bits | Bias   | Correlation |
|------|--------|-------------|
| 2    | +0.001 | 0.80        |
| 3    | +0.000 | 0.93        |
| 4    | +0.000 | 0.98        |

Near-zero bias at all bit-widths confirms the QJL correction works. Correlation of 0.98 at 4-bit means estimated inner products track true values closely.

**Needle-in-Haystack Retrieval**: 9/9 exact retrieval across all bit-widths (2, 3, 4) and sequence lengths (512, 2048, 8192).

### Real Model Validation (Qwen2.5-3B-Instruct)

KV cache captured from a real forward pass on an RTX 5080 (16 GB), then compressed with TurboQuant.

**Compression Ratios** (consistent across all context lengths):

| Config          | KV Cache Size (8K ctx) | Compression |
|-----------------|------------------------|-------------|
| FP16 (baseline) | 289 MB                | 1.0x        |
| TurboQuant 4-bit | 76 MB                 | 3.8x        |
| TurboQuant 3-bit | 58 MB                 | 5.0x        |
| TurboQuant 2-bit | 40 MB                 | 7.3x        |

At 3-bit, 289 MB becomes 58 MB. On a 16 GB GPU, that is the difference between fitting ~8K context and fitting ~40K.

**Attention Score Accuracy** (averaged across all 36 layers, 2 KV heads per layer = 72 total checks):

| Config  | Context | Cosine Sim | Top-1 Match | Top-5 Match |
|---------|---------|------------|-------------|-------------|
| TQ-4bit | 2K      | 0.9989     | 85%         | 96%         |
| TQ-4bit | 4K      | 0.9986     | 92%         | 94%         |
| TQ-4bit | 8K      | 0.9983     | 86%         | 96%         |
| TQ-3bit | 2K      | 0.9961     | 85%         | 94%         |
| TQ-3bit | 4K      | 0.9955     | 75%         | 88%         |
| TQ-3bit | 8K      | 0.9945     | 86%         | 94%         |
| TQ-2bit | 2K      | 0.9897     | 63%         | 83%         |
| TQ-2bit | 4K      | 0.9878     | 65%         | 85%         |
| TQ-2bit | 8K      | 0.9851     | 71%         | 89%         |

**Metric definitions:**

- **Cosine Similarity**: Similarity between the full attention score vectors (compressed vs. original). 0.995 means 99.5% similarity in the overall attention pattern.
- **Top-1 Match**: Fraction of layer-head combinations where the most-attended token is unchanged after compression.
- **Top-5 Match**: Fraction where the true most-attended token remains in the top 5 after compression.

**Key observations:**

- Cosine similarity is stable across context lengths (0.998 at 4-bit regardless of 2K or 8K).
- 3-bit is the practical sweet spot: 5x compression with 99.5% attention fidelity.
- 2-bit is usable but aggressive -- 66% top-1 match means the model occasionally attends to different tokens.
- The paper's "zero accuracy loss" claim at 3.5 bits is plausible given these numbers.

## vLLM Integration

TurboQuant includes a vLLM plugin that compresses the KV cache during inference. The plugin registers via the `vllm.general_plugins` entry point and is activated automatically when the package is installed.

### How It Works

1. On startup, the plugin monkey-patches `FlashAttentionImpl.forward` in vLLM.
2. During **prefill**, standard FP16 attention runs unmodified. The KV cache is written normally.
3. During **decode**, the patched attention compresses cached keys using TurboQuant and computes attention scores via the asymmetric inner product estimator (no decompression step). Values are compressed with MSE-only quantization and decompressed for the weighted sum.

This means prefill latency is unaffected. The compression overhead applies only to decode steps, where it is offset by the reduced memory footprint allowing larger batch sizes.

### Configuration

All configuration is via environment variables:

| Variable                     | Default | Description                                |
|------------------------------|---------|--------------------------------------------|
| `VLLM_TURBOQUANT_ENABLED`   | `1`     | Set to `0` to disable the plugin           |
| `VLLM_TURBOQUANT_BITS`      | `3`     | Quantization bit-width (2, 3, or 4)        |
| `VLLM_TURBOQUANT_HEAD_DIM`  | `256`   | Attention head dimension                   |
| `VLLM_TURBOQUANT_NUM_KV_HEADS` | `2`  | Number of KV heads (for GQA models)        |
| `VLLM_TURBOQUANT_NUM_Q_HEADS`  | `16` | Number of query heads                      |
| `VLLM_TURBOQUANT_LAYERS`    | *(auto)* | Comma-separated layer indices to compress |
| `VLLM_TURBOQUANT_SEED`      | `42`    | Random seed for rotation matrices          |

### Production Setup: Qwen3.5-35B-A3B on RTX 5080 (16 GB)

This is the tested production configuration for running Qwen3.5-35B-A3B with 131K context on a single RTX 5080 16 GB. See [`serve.sh`](serve.sh) for the full script.

```bash
# Reduce CUDA memory fragmentation (required for tight VRAM budgets)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export VLLM_TURBOQUANT_ENABLED=1
export VLLM_TURBOQUANT_BITS=3
export VLLM_TURBOQUANT_HEAD_DIM=256
export VLLM_TURBOQUANT_NUM_KV_HEADS=2
export VLLM_TURBOQUANT_NUM_Q_HEADS=16

vllm serve cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit \
  --port 8082 \
  --enforce-eager \
  --gpu-memory-utilization 0.92 \
  --cpu-offload-gb 20 \
  --max-model-len 131072 \
  --max-num-seqs 2 \
  --trust-remote-code \
  --served-model-name Qwen3.5-35B-A3B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

**Memory budget:**

| Component | GPU | CPU |
|-----------|-----|-----|
| Model weights (AWQ 4-bit) | ~2.5 GiB | ~20 GiB offloaded |
| KV cache (TQ 3-bit compressed) | ~12 GiB | — |
| Compression headroom | ~0.5 GiB | — |
| **Total** | **~15 GiB** | **~20 GiB RAM** |

**Key points:**
- `--gpu-memory-utilization 0.92` (not 0.95) leaves headroom for TurboQuant's compression temporaries. At 0.95, prefill can OOM during the quantizer's broadcast operation.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces CUDA memory fragmentation.
- `--cpu-offload-gb 20` moves most model weights to CPU RAM, freeing GPU VRAM for KV cache.
- `--enforce-eager` is required for the hybrid Mamba+Attention architecture.
- Only 10 of 40 layers use full attention (the rest are GDN/linear) — TurboQuant compresses only those 10 layers.

**vLLM 0.18 patch required:** The hybrid Mamba+Attention architecture with CPU offloading triggers a re-initialization assertion in vLLM 0.18. Change the assert to a warning in `vllm/v1/worker/gpu_model_runner.py`. See [vllm-project/vllm#18298](https://github.com/vllm-project/vllm/pull/18298).

### Other Examples

```bash
# Small model, default settings (head_dim=128)
VLLM_TURBOQUANT_HEAD_DIM=128 \
VLLM_TURBOQUANT_BITS=3 \
  vllm serve Qwen/Qwen2.5-3B-Instruct
```

## Project Structure

```
src/
  turboquant/
    __init__.py              # Package exports
    rotation.py              # Random orthogonal matrix generation
    lloyd_max.py             # Lloyd-Max optimal scalar quantizer solver
    quantizer.py             # TurboQuantMSE (Stage 1), TurboQuantProd (Stage 1+2)
    kv_cache.py              # TurboQuantKVCache wrapper
    compressor.py            # Production compressors for real model tensors
    vllm_plugin/
      __init__.py            # Plugin entry point (vllm.general_plugins)
      config.py              # Environment-based configuration
      codebook.py            # Cached codebook solver
      compressor.py          # GPU-optimized compressors
      shadow_cache.py        # Compressed paged KV block storage
      decode_attention.py    # Asymmetric decode attention kernel
      patch.py               # FlashAttentionImpl monkey-patch
tests/                       # pytest test suite
examples/
  validate_model.py          # Real model validation (Qwen2.5-3B-Instruct)
  run_inference.py           # vLLM inference test
pyproject.toml               # Build configuration and dependencies
```

### Module Details

**`lloyd_max.py`** -- Solves the Lloyd-Max optimal quantizer for the coordinate distribution that arises after random rotation of unit vectors. Uses numerical integration (SciPy) to find centroids that minimize MSE. Codebooks are precomputed once and reused.

**`quantizer.py`** -- Core algorithm. `TurboQuantMSE` implements Stage 1 (rotation + quantization). `TurboQuantProd` adds Stage 2 (QJL residual correction) and provides the unbiased inner product estimator.

**`compressor.py`** -- Production compressors that handle real model tensors (normalization, dtype conversion, asymmetric score computation). `TurboQuantCompressorV2` compresses key vectors and supports `asymmetric_attention_scores()` for computing attention directly from compressed data. `TurboQuantCompressorMSE` compresses value vectors with MSE-only quantization.

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

To run a specific test file:

```bash
pytest tests/test_quantizer.py -v
```

**Note:** The `validate` extra (`bitsandbytes`) requires Linux with CUDA. It will fail on CPU-only machines and most Windows setups.

The model validation example requires a CUDA GPU with at least 6 GB VRAM:

```bash
python examples/validate_model.py
```

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- [QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead](https://arxiv.org/abs/2406.03482) -- the 1-bit residual correction technique
- [PolarQuant: Quantizing KV Caches with Polar Transformation](https://arxiv.org/abs/2502.02617) -- related approach using polar coordinates
- [QJL Reference Implementation](https://github.com/amirzandieh/QJL) -- original CUDA implementation
- [PolarQuant Reference Implementation](https://github.com/ericshwu/PolarQuant)

## License

MIT
