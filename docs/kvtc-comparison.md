# KVTC vs TurboQuant: Competitive Analysis

**Date:** 2026-03-31
**Source:** KVTC paper (arXiv:2511.01815, ICLR 2026, NVIDIA)

---

## Key Finding: Different Races

**KVTC = storage compression** (offline, between turns/offload)
**TurboQuant = inference compression** (online, during generation)

They are complementary, not competing.

---

## Technical Comparison

| Dimension | KVTC (NVIDIA) | TurboQuant (Ours) |
|-----------|--------------|-------------------|
| **Target** | KV cache storage/transfer | Online inference acceleration |
| **Basis** | Cross-layer cross-head PCA (data-dependent) | Random WHT rotation (data-oblivious) |
| **Calibration** | Required (~10 min/model on H100) | None (zero-shot) |
| **Bit allocation** | DP-optimized variable {0,2,4,8} per PC group | Fixed per-layer (uniform or adaptive) |
| **Entropy coding** | DEFLATE via nvCOMP (~1.23x extra) | None |
| **Compression ratio** | 9-88x (kvtc_8x to kvtc_64x) | 2.56x (TQKV_6) to 6x (3-bit) |
| **Runtime behavior** | Decompress before attention (full KV in memory during decode) | Compute attention directly on compressed KV |
| **Inference speedup** | No (storage only; 8x TTFT vs recompute) | Yes (reduced memory bandwidth during decode) |
| **Hardware dependency** | nvCOMP (NVIDIA-only for GPU DEFLATE) | Platform-agnostic (Apple Silicon, CUDA) |
| **Open source** | No | Yes (Apache-2.0) |
| **PCA storage overhead** | 2.4-8.7% of model parameters | None |

---

## Quality Comparison (Llama 3.1 8B)

| Method | CR | GSM8K | MMLU | LITM | RULER-VT |
|--------|-----|-------|------|------|----------|
| Vanilla | 1x | 56.8 | 60.5 | 99.4 | 99.8 |
| FP8 | 2x | 55.2 | 60.1 | 99.4 | 99.1 |
| KIVI 2-bit | 5x | 52.8 | 59.6 | 88.8 | 98.9 |
| GEAR 2-bit | 5x | 52.8 | 59.6 | 96.9 | 99.8 |
| **kvtc_8x** | 9-10x | 57.0 | 59.8 | 99.3 | 99.1 |
| **kvtc_16x** | 18-22x | 56.9 | 60.1 | 99.3 | 99.3 |
| kvtc_32x | 34-44x | 57.8 | 60.6 | 99.1 | 98.9 |
| **TurboQuant TQKV_6** | 2.56x | PPL +0.07% | — | — | — |

Note: KVTC at 10x beats KIVI/GEAR at 5x on every metric. But KVTC doesn't accelerate inference — only reduces storage.

---

## KVTC Latency (H100, Mistral NeMo 12B)

| Operation | batch=8, ctx=8K | batch=2, ctx=16K |
|-----------|----------------|-----------------|
| Compress | 379ms | 194ms |
| Decompress | 267ms | 143ms |
| vs Recompute | 3098ms | — |
| **TTFT improvement** | **8x** | — |

---

## Competitive Positioning Update

### Our real competitors (online KV quantization):
- **KIVI** (ICML 2024): 2-bit, 2.6x compression — quality degrades on LITM (88.8 vs 99.4)
- **GEAR** (2024): 2-bit + low-rank correction — better than KIVI but still 5x max
- **KVQuant** (NeurIPS 2024): 3-4 bit, per-channel — strong but complex

### Where we win:
- TurboQuant TQKV_6 at 2.56x is **lossless** (PPL +0.07%)
- At same 5x compression (TQKV_4), we should beat KIVI/GEAR on quality
- Data-oblivious: no calibration needed
- Platform-agnostic: runs on Apple Silicon

### Where KVTC wins:
- Much higher compression ratios (20-88x) for storage
- Cross-layer PCA captures global redundancy we miss
- But: doesn't help during inference, needs NVIDIA, needs calibration, not open-source

---

## Strategic Implications

1. **TurboQuant and KVTC are a natural two-layer stack:**
   - Layer 1 (online): TurboQuant compresses KV during inference → reduces memory bandwidth
   - Layer 2 (offline): KVTC-style PCA+entropy compresses stored KV between turns → reduces storage

2. **Our competitive moat is online inference compression** — this is where KIVI/GEAR are weak and KVTC doesn't play.

3. **To increase compression ratio while staying online**, we should pursue:
   - Asymmetric K/V (D4): K=6-bit, V=4-bit → 2.4x → 3x with GQA
   - Adaptive layer (D1): outlier layers FP16, normal layers 4-bit → 3-4x
   - Eviction integration (D2): important tokens high-bit, rest low-bit → 5-10x
   - These stack: GQA(8x) × asymmetric+adaptive(3-4x) × eviction(2-3x) = **50-100x total**
