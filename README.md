# TurboQuant

[![CI](https://github.com/sparten0614-cpu/turboquant/actions/workflows/ci.yml/badge.svg)](https://github.com/sparten0614-cpu/turboquant/actions/workflows/ci.yml)

Two-stage KV cache compression for LLM inference. Independent reproduction of the Google Research paper (ICLR 2026).

**Stage 1 (PolarQuant):** Random rotation + Lloyd-Max scalar quantization on the unit sphere.
**Stage 2 (QJL):** 1-bit random projection for unbiased inner-product correction of the quantization residual.

## Results

### Perplexity (WikiText-2, prefill-then-compress)

| Model | Bit-width | PPL | Δ PPL | Compression |
|-------|-----------|-----|-------|-------------|
| TinyLlama-1.1B | 4-bit | 5.94 | +10.3% | 4.0x |
| Sheared-LLaMA-1.3B | 4-bit | 7.07 | -2.8% | 4.0x |
| Qwen2.5-3B | 6-bit | 4.71 | +16.0% | 2.6x |

Optimal bit-width depends on the model's KV cache norm distribution. Models with large K norms (Qwen: max 196) need more bits than Llama-family models (max ~30).

## Install

```bash
# Core (numpy + scipy only)
pip install -e .

# With HuggingFace integration (torch + transformers)
pip install -e ".[hf]"

# With dev dependencies (pytest)
pip install -e ".[dev]"

# Everything
pip install -e ".[all]"
```

## Quick Start

```python
from turboquant import TurboQuantConfig, TurboQuantCompressor
import numpy as np

# Create compressor (4-bit, head_dim=128)
config = TurboQuantConfig(head_dim=128, total_bits=4)
compressor = TurboQuantCompressor(config)

# Compress a KV vector
x = np.random.randn(128)
compressed = compressor.compress(x)
x_hat = compressor.decompress(compressed)

# Or compute inner product directly (faster, no full decompression)
q = np.random.randn(128)
ip = compressor.inner_product(q, compressed)
```

### HuggingFace Integration

```python
from turboquant.hf_integration import TurboQuantCache
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
cache = TurboQuantCache(total_bits=4, head_dim=64)
output = model.generate(input_ids, past_key_values=cache)
```

## Benchmarks

```bash
# TinyLlama (4-bit, recommended)
python benchmarks/perplexity.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --tokens 384 --context 256 --bits 3 4 --device mps

# Qwen2.5-3B (6-bit required for this model)
python benchmarks/perplexity.py --model Qwen/Qwen2.5-3B \
    --tokens 160 --context 128 --bits 4 5 6 --device mps

# Diagnose a model's KV cache properties
python benchmarks/diagnose_qwen.py --model Qwen/Qwen2.5-3B --tokens 256
```

## Tests

```bash
# All tests (~20s for core, ~3min with HF integration)
pytest tests/ -v

# Core tests only (no torch/transformers needed)
pytest tests/test_core.py tests/test_qjl_theorem2.py tests/test_attention.py -v
```

## Architecture

```
turboquant/
├── turboquant.py      # Core compressor: compress/decompress/inner_product
├── codebook.py        # Lloyd-Max optimal scalar quantizer
├── rotation.py        # WHT-based fast random rotation (O(d·log d))
├── bitpack.py         # Bit packing for 2-6 bit formats
└── hf_integration.py  # Drop-in HuggingFace Cache replacement
```

### Compression Pipeline

```
Input x ∈ ℝ^d
  │
  ├─ Stage 1 (PolarQuant, b-1 bits/coord):
  │    x̂ = x/‖x‖  →  Rotate(x̂)  →  Lloyd-Max quantize  →  indices + ‖x‖
  │
  └─ Stage 2 (QJL, 1 bit/coord):
       r = x - x_mse  →  sign(S·r)  →  1-bit signs + ‖r‖
```

Total storage: b bits/coordinate + 2×fp16 scalars (‖x‖, ‖r‖) per vector.

## Bit-Width Selection

| K norm p99 | Recommended | Compression |
|------------|-------------|-------------|
| < 50 | 4-bit | 4.0x |
| 50-100 | 5-bit | 3.0x |
| > 100 | 6-bit | 2.6x |

## Adaptive Layer Selection

Some models have a few layers with extreme Key magnitudes (outlier layers) that dominate quantization error. For example, Qwen2.5-3B Layer 0 and 27 have K_max = 92.8, while normal layers are ~12. Uniform quantization across all layers wastes bits on easy layers while destroying quality on hard ones.

**Solution:** Auto-detect outlier layers (K_max > threshold) and keep them at FP16. All other layers use TurboQuant compression.

```
Layer sensitivity profiling (one-time):
  for each layer: compute K_max over calibration data
  if K_max > threshold → mark as outlier → FP16
  else → TurboQuant 6-bit
```

**Results on Qwen2.5-3B (36 layers):**

| Config | Layers skipped | PPL Δ | Extra memory |
|--------|---------------|-------|-------------|
| Uniform 6-bit | 0 | +4.4% | 0% |
| Skip Layer 0 | 1 | +0.1% | ~2.8% |
| Skip Layer 0+27 | 2 | **+0.04%** | ~5.6% |

Skipping just 2 of 36 layers recovers 99% of the quality loss, at 5.6% extra memory.

**Comparison with prior work:**
- *KVTuner (ICML 2025)* uses complex multi-objective optimization to search per-layer bit allocations. Our approach is a single threshold check — same effect, simpler implementation.
- *KVQuant (NeurIPS 2024)* isolates 1% outlier values per vector into sparse format. Our approach operates at layer granularity — coarser but zero per-vector overhead.

## License

Apache-2.0
