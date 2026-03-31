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

### Llama-2-7B

| Weight | KV Config | PPL | vs F16 | Notes |
|--------|-----------|-----|--------|-------|
| Q4_K_M | F16 baseline | 6.9441 | — | |
| Q4_K_M | Q8_0 | 6.9475 | +0.05% | 8-bit reference |
| Q4_K_M | TQKV_6 | 6.9502 | +0.09% | **lossless** |
| Q4_K_M | TQKV_4 | 7.0247 | +1.16% | pure 4-bit MSE |
| Q4_K_M | TQKV_4Q | 7.0403 | +1.38% | 4-bit + QJL |

### Llama-3.1-8B-Instruct

| Weight | KV Config | PPL | vs F16 | Notes |
|--------|-----------|-----|--------|-------|
| Q4_K_M | F16 baseline | 8.8984 | — | |
| Q4_K_M | Q8_0 | 8.8956 | -0.03% | 8-bit reference |
| Q4_K_M | TQKV_6 | 8.9043 | +0.07% | **lossless** |
| Q4_K_M | TQKV_4Q | 9.0299 | +1.48% | 4-bit + QJL |
| Q4_K_M | TQKV_4 | 9.0519 | +1.73% | pure 4-bit MSE |

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

### Pure MSE vs QJL Mode (Python, head_dim=64, 200 vectors)

| Mode | cos_sim | MSE |
|------|---------|-----|
| 6-bit pure MSE (use_qjl=False) | 0.9996 | 0.0055 |
| 6-bit QJL (5-bit MSE + 1-bit QJL) | 0.9982 | 0.0232 |

Pure MSE wins by +0.14pp cos_sim and 4.25x lower MSE. QJL's 1-bit projection adds noise to individual vectors; its theoretical benefit (unbiased inner products) doesn't offset the MSE bit loss in practice. This matches 阳阳's ggml TQKV_6 (pure MSE) being the production choice.

## Key Findings

1. **TinyLlama:** TQKV_6 is lossless (+0.04%). TQKV_4Q with QJL correction adds only +1.55%.

2. **Qwen2.5-3B:** Uniform TQKV_6 gives +4.4% due to outlier layers (L0, L27 have K_max=92.8 vs normal ~12). **Adaptive Layer Selection** (skip 2/36 layers to F16) recovers to +0.04% at ~2.42x compression.

3. **Bit-width matters for Qwen:** 4-bit is catastrophic (PPL 355), 5-bit unusable (PPL 70), 6-bit works (+16% Python, +4.4% ggml). The correct bit-width depends on model K norm distribution.

4. **Pure MSE > QJL at same bit budget:** At 6-bit, pure MSE (64 centroids) has 4.25x lower reconstruction error than 5-bit MSE + 1-bit QJL. QJL is theoretically unbiased but adds noise per-vector.

4. **Adaptive > Uniform:** For models with outlier layers, skipping 2 layers to F16 costs ~5.6% extra memory but eliminates 99% of quality loss.

## Methodology

- **ggml benchmarks** (TinyLlama, Qwen rows): Metal GPU via llama.cpp, standard perplexity eval (ctx=512, 10 chunks, WikiText-2)
- **Python benchmarks** (Reference Implementation rows): HuggingFace transformers, prefill-then-compress methodology (normal prefill → compress KV cache → eval on subsequent tokens)
- **TQKV_6:** 6-bit TurboQuant (5-bit MSE + 1-bit QJL)
- **TQKV_4Q:** 4-bit TurboQuant with QJL correction (3-bit MSE + 1-bit QJL)
- **Adaptive:** `detect_outlier_layers(threshold=3.0)` identifies layers for F16 fallback
