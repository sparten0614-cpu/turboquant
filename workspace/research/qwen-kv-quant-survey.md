# KV Cache Quantization Survey: Qwen Outlier Layer Problem

**Date:** 2026-03-31
**Author:** 宁宁
**Context:** TurboQuant 6-bit on Qwen2.5-3B gives +3.5% PPL. Root cause: extreme outlier layers (Layer 0/27, K_max=92.8 vs normal ~12), NOT block size mismatch (block=64 MSE < block=128 MSE confirmed by 阳阳).

---

## Problem Analysis

### Empirical Evidence (阳阳's KV dump)
- Block=64 MSE < block=128 MSE (0.72x) — block size is NOT the problem
- Layer 0 and 27: K_max = 92.8 (vs normal layers ~12) — 7.7x magnitude difference
- RoPE dim(i, i+64) correlation < 0.05 — NOT split-half layout, no RoPE pairing issue
- Our Qwen diagnostic: K norm p99=196, Q·K/sqrt(d) error mean=37.7

### Core Issue
A few outlier layers dominate the total quantization error. Uniform 6-bit across all layers wastes bits on easy layers while starving hard layers.

---

## Paper Survey (10 papers, 2024-2026)

### 1. KIVI — Asymmetric 2-bit KV Cache (ICML 2024)

**arXiv:2402.02750** — Liu et al.

**Method:** Keys quantized per-channel (outlier channels dominate), Values quantized per-token (smoother distribution). Asymmetric, tuning-free.

**Key insight:** K/V need different treatment. Keys have per-channel outliers; Values are per-token smooth. PolarQuant treats both identically — suboptimal.

**PPL:** LLaMA-7B 2-bit: "almost same quality", 2.6x memory reduction, 4x batch size.

**Relevance to our problem:** Per-channel Key quantization inherently respects full head_dim, avoids block fragmentation. Suggests TurboQuant should consider asymmetric K/V strategies.

---

### 2. KVQuant — 10M Context Length (NeurIPS 2024)

**arXiv:2401.18079** — Hooper et al. (UC Berkeley)

**Method:** Four-part: (1) per-channel Key quantization (+3.82 PPL improvement for 3-bit LLaMA-7B), (2) **pre-RoPE Key quantization** (+0.82 PPL improvement), (3) per-layer sensitivity-weighted Lloyd-Max codebooks via diagonal Fisher information, (4) per-vector dense-and-sparse outlier isolation (1% outliers in CSR format).

**Key insight:** **Per-layer sensitivity-weighted quantization** — different layers get different codebooks based on Fisher information. Also, **outlier isolation** (1% sparse) handles exactly our Layer 0/27 problem: keep extreme values at full precision, quantize the rest.

**PPL (Wikitext-2):**

| Model | FP16 | 3-bit+1% sparse | 2-bit+1% sparse |
|-------|------|-----------------|-----------------|
| LLaMA-7B | 5.68 | 5.75 (+0.07) | 6.01 (+0.33) |
| LLaMA-13B | 5.09 | 5.14 (+0.05) | 5.36 (+0.27) |
| LLaMA-65B | 3.53 | 3.57 (+0.04) | 3.70 (+0.17) |

**Relevance:** ★★★★★ — Outlier isolation and per-layer sensitivity are directly applicable to our Layer 0/27 problem. Could keep outlier layers at 8-bit while compressing normal layers to 4-bit.

---

### 3. Coupled Quantization (CQ) — 1 Bit Per Channel (NeurIPS 2024)

**NeurIPS 2024** — Zhang et al.

**Method:** Proves distinct channels within KV embeddings are statistically dependent — joint entropy < sum of marginal entropies. Couples multiple channels for joint encoding.

**Key insight:** Theoretical proof that independent block quantization is suboptimal when channels are correlated. However, 阳阳's data shows block=64 actually has LOWER MSE than block=128, and RoPE dim correlation < 0.05, suggesting Qwen's channels are less correlated than CQ assumes. **This paper's premise may not apply to our specific case.**

**PPL:** 1-bit per channel, 1.4-3.5x throughput improvement.

**Relevance:** ★★☆☆☆ — Theoretically interesting but 阳阳's empirical data contradicts the cross-block correlation hypothesis for Qwen.

---

### 4. KVTC — Transform Coding (ICLR 2026)

**arXiv:2511.01815** — Staniszewski, Lancucki (NVIDIA)

**Method:** Three-stage: (1) PCA-based decorrelation across full head_dim, (2) **DP-based adaptive bit allocation** across principal components (high-variance PCs get more bits, low-variance get 0), (3) entropy coding.

**Key insight:** **Adaptive bit allocation via dynamic programming** — instead of uniform bits per dimension, allocate based on variance. For outlier layers where a few dimensions have extreme variance, those get more bits while the rest get fewer. This is a more principled version of "protect outlier layers."

**Compression:** Up to 20x while maintaining accuracy. Tested on Llama 3, Mistral NeMo, R1-Qwen 2.5.

**Relevance:** ★★★★☆ — DP bit allocation could allocate more bits to the outlier dimensions in Layer 0/27 specifically. PCA decorrelation is data-dependent (needs calibration) but more targeted than random WHT.

---

### 5. KVTuner — Layer-Wise Mixed Precision (ICML 2025)

**arXiv:2502.04420** — Li et al.

**Method:** Multi-objective optimization for **per-layer** K/V precision pairs (e.g., K4V2, K8V4). Uses relative attention output error as sensitivity metric. Intra-layer pruning + inter-layer clustering to reduce search space.

**Key insight:** **Keys are 2x more sensitive than Values** — K4V2 gives 0.453 relative error while K2V4 gives 0.892. And critically: **Qwen2.5-7B with uniform KV4 → PPL 220.83 (broken!), but mixed precision ~3.25-bit average → PPL 9.60 (near-lossless).** This directly confirms our Qwen problem: uniform quantization is catastrophic, but per-layer adaptation fixes it.

**PPL (Wikitext-2, group_size=128):**

| Model | FP16 | KV4 (uniform) | KVTuner mixed ~3.25-bit |
|-------|------|---------------|------------------------|
| Llama-3.1-8B | 9.95 | 9.99 | ~9.97 |
| **Qwen2.5-7B** | **9.56** | **220.83** ❌ | **9.60** ✅ |

**Relevance:** ★★★★★ — **Strongest evidence that per-layer adaptation is the solution for Qwen.** Uniform 4-bit destroys Qwen, but mixed precision recovers it completely. Our Layer 0/27 outliers likely correspond to KVTuner's "sensitive layers."

---

### 6. RotateKV — Outlier-Aware Rotation (IJCAI 2025)

**arXiv:2501.16383** — Su et al.

**Method:** (1) **Outlier-aware channel reordering** before WHT — adapts rotation to varying outlier distributions, (2) **Pre-RoPE grouped-head rotation** — rotates across multiple KV heads jointly before RoPE, (3) attention-sink-aware quantization for initial tokens.

**Key insight:** Standard WHT (what PolarQuant uses) doesn't account for the specific outlier structure. RotateKV reorders channels so outliers are optimally distributed before rotation. For GQA models like Qwen (2 KV heads), **cross-head rotation** leverages statistical structure between heads.

**PPL:** LLaMA-2-13B 2-bit: <0.3 PPL degradation. 3.97x memory reduction.

**Relevance:** ★★★★☆ — Outlier-aware rotation directly addresses Layer 0/27's extreme K values. Grouped-head rotation is relevant for Qwen's GQA (2 KV heads).

---

### 7. KVLinC — Hadamard + Linear Correction (Oct 2025)

**arXiv:2510.05373** — Saxena, Roy (Purdue)

**Method:** Channel-wise Key quantization + Hadamard-transformed Value quantization + lightweight linear correction adapters (rank D=256, <1% parameter overhead) that compensate for Key quantization errors.

**Key insight:** **Learned correction adapters** — after quantization introduces error, a small adapter fixes systematic distortion. Constant memory w.r.t. sequence length. **Qwen2.5 specific results available.** Also: "QuaRot fails to produce meaningful results on Qwen" — rotation-based methods struggle with Qwen.

**PPL (2-bit, group_size=128, Wikitext-2):**

| Model | FP16 | KIVI | KVLinC |
|-------|------|------|--------|
| Qwen2.5-1.5B | 9.3 | 16.5 | 13.0 |
| **Qwen2.5-3B** | **8.0** | **9.7** | **8.9** |
| Qwen2.5-7B | 6.8 | 11.2 | 10.5 |
| Llama-3.1-8B | 6.2 | 7.8 | 7.1 |

**Relevance:** ★★★☆☆ — Qwen-specific numbers valuable for comparison. Linear correction is interesting but adds training requirement. Warning that rotation-based methods (like our PolarQuant) may inherently struggle with Qwen.

---

### 8. SKVQ — Channel Reordering (COLM 2024)

**arXiv:2405.06219** — Duanmu et al.

**Method:** KMeans-based channel reordering — group channels with similar distributions before quantization. Permutation fused into projection weights (zero overhead). Recent tokens in sliding window at full precision.

**Key insight:** Data-driven channel grouping is a lightweight alternative to rotation. For outlier layers, similar-magnitude channels get grouped together, reducing within-group variance.

**PPL (Llama-2-13B, Wikitext-2):** 2-bit: 4.87 (vs FP16 4.57, +0.30).

**Relevance:** ★★☆☆☆ — Channel reordering is less relevant now that block size is ruled out as root cause. Sliding window for recent tokens could complement our approach.

---

### 9. ZipCache — Token-Level Mixed Precision (NeurIPS 2024)

**arXiv:2405.14256** — He et al.

**Method:** Attention-score-based token saliency. Salient tokens → 4-bit, non-salient → 2-bit. Channel-wise Keys, token-wise Values.

**Key insight:** Token-level mixed precision is orthogonal to layer-level. Could combine: outlier layers at high-bit + salient tokens at high-bit.

**Results:** Mistral-7B GSM8k: 4.98x compression, 0.38% accuracy drop.

**Relevance:** ★★☆☆☆ — Complementary approach, not primary solution for outlier layers.

---

### 10. QServe — W4A8KV4 System (MLSys 2025)

**arXiv:2405.04532** — Lin et al. (MIT Han Lab)

**Method:** **SmoothAttention** — migrates quantization difficulty from KV cache to query. Adjusts query computation to be less sensitive to KV errors.

**Key insight:** Instead of only fixing KV side, also adjust queries. If query is smoothed, the same KV quantization error has less impact on attention output.

**Results:** Qwen1.5-72B: 2.4x throughput on A100, 3.5x on L40S.

**Relevance:** ★★★☆☆ — SmoothAttention is a novel angle: reduce error impact rather than error magnitude. Could complement PolarQuant.

---

## Synthesis: Recommendations for TurboQuant

### Root Cause (confirmed)
Outlier layers (Layer 0/27, K_max=92.8) dominate quantization error. Uniform bit allocation wastes budget.

### Priority Solutions

**P0: Per-layer adaptive bit allocation (KVTuner-style)**
- KVTuner proves this works for Qwen: uniform KV4 → PPL 220, mixed ~3.25-bit → PPL 9.60
- Allocate 8-bit to Layer 0/27 (outlier layers), 4-bit to normal layers
- Average bit-width stays ~6, but error drops dramatically
- Implementation: sensitivity profiling pass (one-time), then per-layer config

**P1: Outlier isolation (KVQuant-style)**
- Keep top 1% outlier values per vector at full precision (sparse format)
- Rest quantized normally
- Directly handles K_max=92.8 without changing the whole layer's bit-width

**P2: Asymmetric K/V allocation (KVTuner + "More for Keys")**
- Keys are 2x more sensitive than Values
- Use K6V4 instead of KV6 — same average budget, better quality
- Matches KIVI's per-channel K / per-token V strategy

**P3: Outlier-aware rotation (RotateKV)**
- Reorder channels before WHT to better distribute outliers
- Grouped-head rotation for Qwen's GQA (2 KV heads)

### Not Recommended
- Pre-RoPE quantization — requires architecture changes, not applicable to post-prefill compression
- Learned correction adapters (KVLinC) — adds training requirement, against our "tuning-free" design
- Block size increase — block=64 already better than block=128 per empirical data

---

## Key Numbers for Reference

| Paper | Model | Method | Bits | PPL | Δ PPL |
|-------|-------|--------|------|-----|-------|
| KVTuner | Qwen2.5-7B | Uniform KV4 | 4 | 220.83 | +2210% ❌ |
| KVTuner | Qwen2.5-7B | Mixed ~3.25 | 3.25 | 9.60 | +0.4% ✅ |
| KVLinC | Qwen2.5-3B | 2-bit+adapter | 2 | 8.9 | +11.3% |
| KIVI | Qwen2.5-3B | 2-bit asym | 2 | 9.7 | +21.3% |
| KVQuant | LLaMA-7B | 3-bit+1%sparse | 3 | 5.75 | +1.2% |
| **Ours** | **Qwen2.5-3B** | **TurboQuant 6-bit** | **6** | **4.71** | **+16.0%** |
| **Ours** | **TinyLlama** | **TurboQuant 4-bit** | **4** | **5.94** | **+10.3%** |

---

## References

1. KIVI — arXiv:2402.02750 (ICML 2024)
2. KVQuant — arXiv:2401.18079 (NeurIPS 2024)
3. Coupled Quantization — NeurIPS 2024
4. KVTC — arXiv:2511.01815 (ICLR 2026)
5. KVTuner — arXiv:2502.04420 (ICML 2025)
6. RotateKV — arXiv:2501.16383 (IJCAI 2025)
7. KVLinC — arXiv:2510.05373 (Oct 2025)
8. SKVQ — arXiv:2405.06219 (COLM 2024)
9. ZipCache — arXiv:2405.14256 (NeurIPS 2024)
10. QServe — arXiv:2405.04532 (MLSys 2025)
