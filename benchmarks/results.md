# TurboQuant Benchmark Results

**Date:** 2026-03-31
**Platform:** Apple Silicon (Metal GPU), llama.cpp
**Dataset:** WikiText-2 validation
**Eval config:** ctx=512, 10 chunks
**Quantization:** Post-prefill KV cache compression

## Results

### TinyLlama 1.1B

| Weight | KV Config | PPL | vs F16 | Compression | Notes |
|--------|-----------|-----|--------|-------------|-------|
| Q4_K_M | F16 baseline | 25.54 | — | 1x | |
| Q4_K_M | TQKV_6 | 25.55 | +0.04% | 2.56x | near-lossless |
| Q4_K_M | TQKV_4Q | 25.94 | +1.55% | 2.9x | with QJL correction |

### Qwen2.5-3B

| Weight | KV Config | PPL | vs F16 | Compression | Notes |
|--------|-----------|-----|--------|-------------|-------|
| Q4_K_M | F16 baseline | 10.75 | — | 1x | |
| Q4_K_M | TQKV_6 | 11.23 | +4.4% | 2.56x | outlier layers hurt |
| Q4_K_M | TQKV_6 + skip L0 | 10.77 | +0.1% | ~2.45x | adaptive, 1 layer F16 |
| Q4_K_M | TQKV_6 + skip L0,27 | 10.75 | +0.04% | ~2.42x | **lossless** |
| Q8_0 | TQKV_6 | 10.91 | +3.5% | 2.56x | higher weight precision |

### Python Reference Implementation

| Model | KV Config | PPL | vs F16 | Notes |
|-------|-----------|-----|--------|-------|
| TinyLlama 1.1B (head_dim=64) | 4-bit | 5.94 | +10.3% | ctx=256, eval=128 |
| Sheared-LLaMA 1.3B (head_dim=128) | 4-bit | 7.07 | -2.8% | lossless |
| Qwen2.5-3B (head_dim=128) | 4-bit | 354.84 | +8631% | broken — norm too large |
| Qwen2.5-3B (head_dim=128) | 5-bit | 70.28 | +1629% | improved but unusable |
| Qwen2.5-3B (head_dim=128) | 6-bit | 4.71 | +16.0% | near-lossless |

## Key Findings

1. **TinyLlama:** TQKV_6 is lossless (+0.04%). TQKV_4Q with QJL correction adds only +1.55%.

2. **Qwen2.5-3B:** Uniform TQKV_6 gives +4.4% due to outlier layers (L0, L27 have K_max=92.8 vs normal ~12). **Adaptive Layer Selection** (skip 2/36 layers to F16) recovers to +0.04% at ~2.42x compression.

3. **Bit-width matters for Qwen:** 4-bit is catastrophic (PPL 355), 5-bit unusable (PPL 70), 6-bit works (+16% Python, +4.4% ggml). The correct bit-width depends on model K norm distribution.

4. **Adaptive > Uniform:** For models with outlier layers, skipping 2 layers to F16 costs ~5.6% extra memory but eliminates 99% of quality loss.

## Methodology

- **ggml benchmarks** (TinyLlama, Qwen rows): Metal GPU via llama.cpp, standard perplexity eval (ctx=512, 10 chunks, WikiText-2)
- **Python benchmarks** (Reference Implementation rows): HuggingFace transformers, prefill-then-compress methodology (normal prefill → compress KV cache → eval on subsequent tokens)
- **TQKV_6:** 6-bit TurboQuant (5-bit MSE + 1-bit QJL)
- **TQKV_4Q:** 4-bit TurboQuant with QJL correction (3-bit MSE + 1-bit QJL)
- **Adaptive:** `detect_outlier_layers(threshold=3.0)` identifies layers for F16 fallback
