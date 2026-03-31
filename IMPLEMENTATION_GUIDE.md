# TurboQuant Implementation Guide

**Comprehensive Technical Breakdown for KV Cache Compression**

Sources:
- TurboQuant (arXiv:2504.19874) — ICLR 2026
- PolarQuant (arXiv:2502.02617) — AISTATS 2026
- QJL (arXiv:2406.03482) — Quantized Johnson-Lindenstrauss Transform

---

## Table of Contents

1. [Overview and Architecture](#1-overview-and-architecture)
2. [Algorithm Pipeline](#2-algorithm-pipeline)
3. [PolarQuant Details](#3-polarquant-details)
4. [QJL Correction](#4-qjl-correction)
5. [Complete TurboQuant Algorithm](#5-complete-turboquant-algorithm)
6. [Bit Packing Format](#6-bit-packing-format)
7. [Dequantization for Attention](#7-dequantization-for-attention)
8. [Key vs Value Treatment](#8-key-vs-value-treatment)
9. [Integration with Attention](#9-integration-with-attention)
10. [Mathematical Formulas Reference](#10-mathematical-formulas-reference)
11. [Hyperparameters](#11-hyperparameters)
12. [Performance Characteristics](#12-performance-characteristics)
13. [Implementation Considerations](#13-implementation-considerations)

---

## 1. Overview and Architecture

TurboQuant is a **two-stage, data-oblivious** vector quantization scheme for compressing KV cache entries in transformer models. "Data-oblivious" means no training, fine-tuning, or dataset-specific calibration is required — the algorithm works on any input distribution.

### Core Idea

Given a KV vector `x ∈ ℝ^d`:

1. **Stage 1 (PolarQuant / MSE-optimal):** Randomly rotate `x`, then apply per-coordinate scalar quantization using a precomputed Lloyd-Max codebook. This uses `(b-1)` bits per coordinate.
2. **Stage 2 (QJL correction):** Compute the residual from Stage 1, project it with a random Gaussian matrix, and take the sign (1 bit per coordinate). This corrects the inner-product bias introduced by MSE-optimal quantization.

Total: `(b-1) + 1 = b` bits per coordinate, plus a single scalar (residual norm `γ`) per vector.

### Why Two Stages?

MSE-optimal quantizers minimize reconstruction error but introduce **bias** in inner product estimation — `E[⟨y, x̃⟩] ≠ ⟨y, x⟩`. For attention computation, we need **unbiased** inner products (attention logits are inner products of queries and keys). The QJL correction eliminates this bias.

---

## 2. Algorithm Pipeline

### Compression (per KV vector `x ∈ ℝ^d`)

```
Input: x ∈ ℝ^d, rotation matrix Π ∈ ℝ^(d×d),
       codebook C = {c_1, ..., c_{2^{b-1}}},
       random projection S ∈ ℝ^(d×d)

Step 1: Rotate
    y ← Π · x

Step 2: Scalar quantize each coordinate (b-1 bits each)
    For j = 1..d:
        idx_j ← argmin_k |y_j - c_k|

Step 3: Compute MSE reconstruction
    For j = 1..d:
        ỹ_j ← c_{idx_j}
    x̃_mse ← Πᵀ · ỹ

Step 4: Compute residual
    r ← x - x̃_mse

Step 5: Store residual norm
    γ ← ‖r‖₂

Step 6: QJL quantize residual (1 bit per coordinate)
    qjl ← sign(S · r)     // d-dimensional vector of {-1, +1}

Output: (idx[1..d], qjl[1..d], γ)
         ↑ (b-1)·d bits    ↑ d bits   ↑ fp16
```

### Decompression (for inner product with query `y`)

```
Input: (idx, qjl, γ), rotation Π, codebook C, projection S, query y

Step 1: MSE dequantize
    ỹ_j ← c_{idx_j}  for j = 1..d
    x̃_mse ← Πᵀ · ỹ

Step 2: QJL dequantize residual
    x̃_qjl ← (√(π/2) / d) · γ · Sᵀ · qjl

Step 3: Final reconstruction
    x̃ ← x̃_mse + x̃_qjl

Output: x̃  (or compute ⟨y, x̃⟩ directly without materializing x̃)
```

---

## 3. PolarQuant Details

PolarQuant is the MSE-optimal component of TurboQuant. There are two equivalent formulations in the literature: the **Cartesian formulation** (used in TurboQuant paper) and the **polar coordinate formulation** (from the PolarQuant paper). Both achieve the same effect.

### 3.1 Cartesian Formulation (TurboQuant)

#### Random Rotation

The rotation matrix `Π ∈ ℝ^(d×d)` is constructed by:

```
G ← random matrix with i.i.d. entries G_{ij} ~ N(0, 1)
Π, R ← QR(G)     // QR decomposition; Π is orthogonal
```

**Key property:** After rotation `y = Π·x`, if `x` is on the unit sphere `S^{d-1}`, each coordinate `y_j` follows the distribution:

```
f_X(x) = [Γ(d/2) / (√π · Γ((d-1)/2))] · (1 - x²)^{(d-3)/2}
```

This is a **Beta-like distribution** supported on `[-1, 1]`. For large `d`, it converges to `N(0, 1/d)`.

**Critical insight:** All coordinates share the **same** marginal distribution and are **nearly independent** in high dimensions. This means a single scalar quantizer works for all coordinates.

#### Lloyd-Max Codebook Construction

The codebook `{c_1, ..., c_{2^b}}` is constructed by solving the continuous 1D k-means problem:

```
minimize  Σ_{i=1}^{2^b}  ∫_{(c_{i-1}+c_i)/2}^{(c_i+c_{i+1})/2}  |x - c_i|² · f_X(x) dx
```

where `f_X` is the coordinate distribution above.

**Implementation:** This is solved via the standard Lloyd-Max algorithm (iterative):

```
Repeat until convergence:
    1. Update boundaries: b_i = (c_i + c_{i+1}) / 2
    2. Update centroids: c_i = E[X | X ∈ [b_{i-1}, b_i]]
                              = ∫_{b_{i-1}}^{b_i} x · f_X(x) dx / ∫_{b_{i-1}}^{b_i} f_X(x) dx
```

**The codebook is precomputed once** for a given dimension `d` and bit-width `b`, then reused for all vectors. Since the coordinate distribution depends only on `d`, this is a one-time cost.

**Practical note:** For large `d` (typical in LLMs: d=64-128 per head), the distribution is well-approximated by `N(0, 1/d)`, so you can compute the Lloyd-Max codebook for Gaussian and scale by `1/√d`.

#### Precomputed Codebook Values

The paper provides MSE distortion values for specific bit-widths on the unit sphere:

| Bit-width b | MSE per coordinate | Notes |
|---|---|---|
| 1 | 0.3634 | |
| 2 | 0.1175 | |
| 3 | 0.03454 | |
| 4 | 0.009497 | |

These are `C(f_X, b)` values — the optimal scalar quantization distortion.

### 3.2 Polar Coordinate Formulation (PolarQuant Paper)

The PolarQuant paper offers an alternative (mathematically equivalent) perspective using polar coordinates.

#### Random Preconditioning

Apply a random orthogonal matrix `S ∈ ℝ^{d×d}` (same as `Π` above):
```
y ← S · x
```

#### Recursive Polar Transform

For `d` a power of 2, transform `y ∈ ℝ^d` into polar coordinates via a recursive algorithm:

```
Polar(y):
    r^(0) ← y

    for ℓ = 1, ..., log₂(d):
        for j = 1, ..., d/2^ℓ:
            ψ_j^(ℓ) ← arctan(r^(ℓ-1)_{2j} / r^(ℓ-1)_{2j-1})
            r_j^(ℓ) ← ‖[r^(ℓ-1)_{2j-1}, r^(ℓ-1)_{2j}]‖₂

    return r^(log₂d), {ψ^(1), ..., ψ^(log₂d)}
```

This produces:
- `d/2` angles at level 1 (range `[0, 2π)`)
- `d/4` angles at level 2 (range `[0, π/2]`)
- ...
- 1 angle at level `log₂(d)`
- 1 final radius `r = ‖x‖₂`

Total: `d - 1` angles + 1 radius.

#### Angle Distributions After Rotation

- **Level 1:** Uniform over `[0, 2π)` — `f(ψ) = 1/(2π)`
- **Level ℓ ≥ 2:** Independent with density:

```
f_ψ^(ℓ)(ψ) = [Γ(2^{ℓ-1}) / (2^{2^{ℓ-1}-2} · Γ(2^{ℓ-2})²)] · sin^{2^{ℓ-1}-1}(2ψ)
```

**Concentration:** Mean `π/4`, variance `O(1/√d)`. Higher levels are increasingly concentrated around `π/4`.

#### Quantizing Angles

Each level uses its own codebook (because the distributions differ):

```
For each level ℓ:
    Codebook: {θ_1^(ℓ), ..., θ_{2^b}^(ℓ)} via k-means on f_ψ^(ℓ)
    For each angle ψ_i^(ℓ):
        j_i ← argmin_k |ψ_i^(ℓ) - θ_k^(ℓ)|
```

#### Dequantization (Inverse Polar Transform)

```
InversePolar(r, {j^(1), ..., j^(log₂d)}, codebooks):
    r^(log₂d) ← [r]

    for ℓ = log₂(d), ..., 1:
        for j = 1, ..., d/2^ℓ:
            θ ← θ_{j_j^(ℓ)}^(ℓ)   // look up centroid
            r^(ℓ-1)_{2j-1} ← r_j^(ℓ) · cos(θ)
            r^(ℓ-1)_{2j}   ← r_j^(ℓ) · sin(θ)

    return S^T · r^(0)    // undo rotation
```

#### Practical Configuration (PolarQuant paper)

- **Recursion depth:** L = 4 (not full `log₂(d)`) for practical balance
- **Bit allocation per level:**
  - Level 1: 4 bits (wider range `[0, 2π)`)
  - Levels 2-4: 2 bits each
- **Effective rate:** ~3.875 bits per coordinate (with fp16 radius overhead amortized)
- **Complexity:** `O(d log d)` per vector

### 3.3 Equivalence

Both formulations achieve the same MSE distortion — they're different decompositions of the same mathematical insight: random rotation induces a known, concentrated distribution on coordinates, enabling efficient scalar quantization without data-dependent normalization.

---

## 4. QJL Correction

### 4.1 The Problem

MSE-optimal quantizers (PolarQuant / Stage 1) minimize `E[‖x - x̃‖²]` but do NOT guarantee `E[⟨y, x̃⟩] = ⟨y, x⟩`. The inner product estimate is **biased**. For attention computation, we need unbiased inner products.

### 4.2 QJL Definition

**Quantization map:**
```
Q_qjl(x) := sign(S · x)
```
where `S ∈ ℝ^{m×d}` is a random matrix with i.i.d. entries `S_{ij} ~ N(0, 1)`.

In TurboQuant, `m = d` (same dimensionality), so each coordinate gets exactly 1 bit.

**Dequantization map:**
```
Q_qjl^{-1}(z) := (√(π/2) / m) · Sᵀ · z
```

### 4.3 Key Properties

**Unbiasedness (Lemma 4):**
```
E[⟨y, Q_qjl^{-1}(Q_qjl(x))⟩] = ⟨y, x⟩
```

Proof sketch: For any `u ~ N(0,1)`, `E[sign(u) · u'] = √(2/π)` when `u, u'` are jointly Gaussian. The `√(π/2)` factor compensates exactly.

**Variance bound:**
```
Var(⟨y, Q_qjl^{-1}(Q_qjl(x))⟩) ≤ (π / (2m)) · ‖y‖₂² · ‖x‖₂²
```

With `m = d`:
```
Var ≤ (π / (2d)) · ‖y‖₂² · ‖x‖₂²
```

### 4.4 The √(π/2) Factor

This comes from the identity: for `g ~ N(0, σ²)`:
```
E[|g|] = σ · √(2/π)
```

When we take `sign(g)` and want to recover `g`, we need to scale by `E[|g|] = σ√(2/π)`, giving the inverse factor `√(π/2)/σ`.

### 4.5 Application to Residuals

In TurboQuant, QJL is applied to the **residual** `r = x - x̃_mse`, not to `x` directly:

```
qjl_bits ← sign(S · r)           // 1 bit per coordinate
γ ← ‖r‖₂                         // scalar, stored in fp16

// Reconstruction of residual contribution:
r̃ ← (√(π/2) / d) · γ · Sᵀ · qjl_bits
```

The `γ` factor is necessary because the QJL estimator assumes unit-norm input; scaling by `γ = ‖r‖₂` handles the actual magnitude.

### 4.6 The Random Matrix S

- **Dimensions:** `d × d` (square, same as head dimension)
- **Entries:** i.i.d. `N(0, 1)`
- **Storage:** In practice, `S` is generated from a fixed random seed (not stored explicitly). The seed must be shared between compression and decompression.
- **One matrix per model** is sufficient (shared across layers/heads), since the theoretical guarantees hold regardless of input distribution.

### 4.7 Why Not Just Use QJL Alone?

QJL alone gives only 1 bit per coordinate with variance `O(1/d)`. The two-stage approach gets the MSE benefit of `(b-1)` bits from PolarQuant, then uses the 1-bit QJL to debias, achieving both low MSE and unbiased inner products at `b` total bits.

---

## 5. Complete TurboQuant Algorithm

### 5.1 Preprocessing (one-time)

```python
# Generate rotation matrix (shared across all vectors)
G = np.random.randn(d, d)        # Gaussian random matrix
Pi, _ = np.linalg.qr(G)          # Orthogonal rotation

# Generate QJL projection (shared, or use seeded RNG)
S = np.random.randn(d, d)        # Gaussian projection matrix

# Compute Lloyd-Max codebook for (b-1) bits
# Distribution: f_X(x) ∝ (1 - x²)^{(d-3)/2}  on [-1, 1]
# For large d: approximately N(0, 1/d)
codebook = lloyd_max(f_X, num_levels=2**(b-1))
# codebook.centroids: array of 2^{b-1} values
# codebook.boundaries: array of 2^{b-1}-1 boundary values
```

### 5.2 Quantization (Algorithm 2 from paper)

```python
def turboquant_compress(x, Pi, S, codebook):
    """
    x: input vector, shape (d,)
    Returns: (indices, qjl_bits, gamma)
    """
    # Stage 1: MSE quantization with (b-1) bits
    y = Pi @ x                              # Rotate
    indices = quantize_scalar(y, codebook)  # Per-coordinate, (b-1) bits each
    y_hat = codebook.centroids[indices]     # Dequantized in rotated space
    x_mse = Pi.T @ y_hat                   # Back to original space

    # Stage 2: QJL on residual
    r = x - x_mse                           # Residual
    gamma = np.linalg.norm(r)               # Residual norm (scalar)
    qjl_bits = np.sign(S @ r)              # 1 bit per coordinate

    return indices, qjl_bits, gamma

def quantize_scalar(y, codebook):
    """Quantize each coordinate to nearest centroid."""
    # Binary search or linear scan over sorted centroids
    indices = np.searchsorted(codebook.boundaries, y)
    return indices  # Each in {0, 1, ..., 2^{b-1} - 1}
```

### 5.3 Dequantization (Algorithm 2 from paper)

```python
def turboquant_decompress(indices, qjl_bits, gamma, Pi, S, codebook):
    """
    Returns: reconstructed vector x̃, shape (d,)
    """
    # Stage 1: MSE reconstruction
    y_hat = codebook.centroids[indices]
    x_mse = Pi.T @ y_hat

    # Stage 2: QJL residual reconstruction
    scale = math.sqrt(math.pi / 2) / d
    x_qjl = scale * gamma * (S.T @ qjl_bits)

    return x_mse + x_qjl
```

### 5.4 Inner Product Without Full Decompression

For attention, you often need `⟨q, k̃⟩` rather than `k̃` itself:

```python
def turboquant_inner_product(q, indices, qjl_bits, gamma, Pi, S, codebook):
    """
    Compute ⟨q, k̃⟩ without fully materializing k̃.
    """
    # MSE component: ⟨q, Πᵀ ỹ⟩ = ⟨Π q, ỹ⟩
    q_rot = Pi @ q
    y_hat = codebook.centroids[indices]
    ip_mse = np.dot(q_rot, y_hat)

    # QJL component: ⟨q, (√(π/2)/d) · γ · Sᵀ · z⟩ = (√(π/2)/d) · γ · ⟨S q, z⟩
    q_proj = S @ q
    ip_qjl = (math.sqrt(math.pi / 2) / d) * gamma * np.dot(q_proj, qjl_bits)

    return ip_mse + ip_qjl
```

This avoids materializing the full `d`-dimensional decompressed vector.

---

## 6. Bit Packing Format

### 6.1 Storage Layout Per Vector

For a `d`-dimensional vector at `b` bits per coordinate:

| Component | Bits | Description |
|---|---|---|
| MSE indices | `(b-1) × d` | Per-coordinate quantization indices |
| QJL signs | `1 × d` | Sign bits from `sign(S · r)` |
| Gamma (γ) | 16 | Residual norm in fp16 |
| **Total** | `b × d + 16` | |

Effective bits per coordinate: `b + 16/d`. For `d = 128`: `b + 0.125`.

### 6.2 Packing Strategy

#### For b-1 = 1 (2-bit total: 1-bit MSE + 1-bit QJL)

```
Pack MSE index (1 bit) and QJL sign (1 bit) into consecutive bits.
Per coordinate: 2 bits
Per uint8: 4 coordinates
Per uint32: 16 coordinates

Layout in uint8: [mse₀|qjl₀|mse₁|qjl₁|mse₂|qjl₂|mse₃|qjl₃]
```

#### For b-1 = 2 (3-bit total: 2-bit MSE + 1-bit QJL)

```
Per coordinate: 3 bits
Pack into uint8: 2 coordinates per byte (6 bits used, 2 wasted)
  OR
Pack 8 coordinates into 3 bytes (24 bits, zero waste):
  byte 0: [idx₀(2)|qjl₀(1)|idx₁(2)|qjl₁(1)|idx₂_hi(2)]
  byte 1: [idx₂_lo(0)|qjl₂(1)|idx₃(2)|qjl₃(1)|idx₄(2)|qjl₄_hi(1)]
  ... etc

Practical approach: store MSE indices and QJL bits in separate arrays:
  mse_packed: uint8 array, 4 indices per byte (2-bit each)
  qjl_packed: uint8 array, 8 signs per byte (1-bit each)
```

#### For b-1 = 3 (4-bit total: 3-bit MSE + 1-bit QJL)

```
Separate storage:
  mse_packed: 3 bits per index → pack 8 indices into 3 bytes
  qjl_packed: 1 bit per sign → 8 signs per byte
```

#### For b-1 = 4 (5-bit total: 4-bit MSE + 1-bit QJL)

```
Separate storage (recommended):
  mse_packed: 4 bits per index → 2 indices per byte (natural nibble packing)
  qjl_packed: 1 bit per sign → 8 signs per byte
```

### 6.3 Recommended Implementation

Separate arrays are simpler and more GPU-friendly:

```
struct CompressedKVCache {
    uint8_t* mse_indices;    // packed (b-1)-bit indices
    uint8_t* qjl_signs;     // packed 1-bit signs
    half*    gamma;          // one fp16 per vector
    // Shapes:
    //   mse_indices: [num_layers, num_heads, seq_len, ceil(d*(b-1)/8)]
    //   qjl_signs:   [num_layers, num_heads, seq_len, ceil(d/8)]
    //   gamma:        [num_layers, num_heads, seq_len]
};
```

---

## 7. Dequantization for Attention

### 7.1 Strategy: Decompress-on-the-fly

TurboQuant **fully decompresses** KV vectors during attention computation. There is no "compressed-domain" attention. The workflow is:

```
For each attention head:
    For each cached token position t:
        k̃_t ← TurboQuant_decompress(compressed_k[t])
        ṽ_t ← TurboQuant_decompress(compressed_v[t])

    // Standard attention with decompressed K, V
    attn_logits = q @ K̃ᵀ / √d
    attn_weights = softmax(attn_logits)
    output = attn_weights @ Ṽ
```

### 7.2 Optimized: Fused Decompression-Matmul

In practice, you fuse decompression with the matrix multiply to avoid materializing full K/V matrices:

```
For computing q @ K̃ᵀ (attention logits):
    For each cached position t:
        logit_t = turboquant_inner_product(q, k_compressed[t])
    // This avoids ever storing the full decompressed K matrix

For computing attn_weights @ Ṽ (attention output):
    Must decompress V vectors, as the weighted sum requires full vectors.
    output = Σ_t  attn_weights[t] · turboquant_decompress(v_compressed[t])
```

### 7.3 GPU Kernel Design

The key kernel fuses:
1. Unpack MSE indices from packed format
2. Look up codebook centroids
3. Apply inverse rotation (Πᵀ multiply)
4. Unpack QJL signs
5. Compute QJL correction term
6. Accumulate into attention output

For **key attention** (logits), the fused kernel computes inner products directly:
```
// Pseudocode for fused K-attention kernel
__global__ void fused_k_attention(q, mse_indices, qjl_signs, gamma,
                                   Pi, S, codebook, output_logits):
    // Each thread block handles one query position against all keys
    q_rot = Pi @ q           // Rotate query once
    q_proj = S @ q           // Project query once

    for t in range(seq_len):
        // MSE inner product
        y_hat = lookup(codebook, mse_indices[t])
        ip_mse = dot(q_rot, y_hat)

        // QJL inner product
        signs = unpack(qjl_signs[t])
        ip_qjl = sqrt(pi/2)/d * gamma[t] * dot(q_proj, signs)

        output_logits[t] = ip_mse + ip_qjl
```

### 7.4 The 4-bit Performance Advantage

The paper reports that **4-bit TurboQuant achieves up to 8x performance increase** over 32-bit unquantized keys on H100 GPUs. This is because:
- 8x less memory bandwidth to read KV cache
- Decompression is compute-bound (cheap) while attention is memory-bandwidth-bound
- The decompression cost is amortized across the memory savings

---

## 8. Key vs Value Treatment

### 8.1 Unified Framework

TurboQuant applies the **same algorithm** to both keys and values. There is no explicit asymmetry in the paper's formulation.

### 8.2 Why Keys Need Unbiased Inner Products

- **Keys** participate in `q · kᵀ` (attention logits). Biased inner products shift attention weights, distorting which tokens get attended to. The QJL correction is **critical** for keys.
- **Values** participate in the weighted sum `Σ attn_weight_t · v_t`. MSE-optimal reconstruction (minimizing `‖v - ṽ‖²`) directly serves this use case. The QJL correction is **less critical** for values but still helps.

### 8.3 Practical Asymmetry (Potential Optimization)

While the paper treats K and V identically, an implementation could:
- Use full TurboQuant (MSE + QJL) for **keys** at `b` bits
- Use MSE-only quantization for **values** at `b` bits (skip QJL)
- This saves the `γ` storage and QJL computation for values

The paper does not explore this, but it's a valid engineering tradeoff since value reconstruction benefits more from MSE minimization than inner-product unbiasedness.

### 8.4 Outlier Handling

The paper mentions "outlier and non-outlier sets" receiving different bit allocations:
- Outlier channels (those with large magnitude) can receive higher bit-width (e.g., 4 bits)
- Non-outlier channels receive lower bit-width (e.g., 2 bits)
- This is applied uniformly to both K and V

---

## 9. Integration with Attention

### 9.1 Where Compression/Decompression Happens

In the transformer forward pass:

```
                    ┌─────────────────────────────────────────┐
                    │          Standard Transformer Layer       │
                    │                                           │
Input x ─→ [Q,K,V projection] ─→ Q, K_new, V_new             │
                    │              │        │                    │
                    │         ┌────┘        └────┐              │
                    │         ▼                   ▼              │
                    │    COMPRESS(K_new)    COMPRESS(V_new)      │
                    │         │                   │              │
                    │         ▼                   ▼              │
                    │    ┌─────────┐        ┌─────────┐         │
                    │    │ KV Cache │        │ KV Cache │        │
                    │    │(compress)│        │(compress)│        │
                    │    └────┬────┘        └────┬────┘         │
                    │         │                   │              │
                    │    DECOMPRESS          DECOMPRESS          │
                    │         │                   │              │
                    │         ▼                   ▼              │
                    │    K_all (full)        V_all (full)        │
                    │         │                   │              │
                    │    ┌────┘                   │              │
                    │    ▼                        ▼              │
                    │  Q @ K_allᵀ/√d ─→ softmax ─→ @ V_all     │
                    │         │                                  │
                    │         ▼                                  │
                    │    Attention Output                        │
                    └─────────────────────────────────────────────┘
```

### 9.2 Timing

- **Prefill phase:** All KV vectors for the prompt are computed, then compressed and stored in the cache.
- **Generation phase:** Each new token produces one new K and one new V vector, which are compressed and appended. All cached KV are decompressed for attention.
- **New tokens during generation** can optionally be stored uncompressed (full precision) since there's only one per step.

### 9.3 The Rotation and Projection Matrices

Both `Π` (rotation) and `S` (QJL projection) are:
- Generated once per model (or once per session with a fixed seed)
- Shared across all layers, heads, and sequence positions
- Stored in GPU memory as constants
- Size: `d × d` each in float16 = `2 × d² × 2` bytes per head dimension

For `d = 128`: each matrix is 32 KB in fp16. This is negligible overhead.

### 9.4 Computational Cost

Per vector compression:
```
Rotation:          O(d²)    — matrix-vector multiply
Scalar quantize:   O(d·b)   — binary search per coordinate
MSE dequantize:    O(d²)    — matrix-vector multiply (for residual)
QJL projection:    O(d²)    — matrix-vector multiply
Sign:              O(d)     — elementwise
Norm:              O(d)     — reduction
Total:             O(d²)    — dominated by matrix multiplies
```

Per vector decompression:
```
Codebook lookup:   O(d)
Inverse rotation:  O(d²)    — Πᵀ multiply
QJL reconstruct:   O(d²)    — Sᵀ multiply
Total:             O(d²)
```

---

## 10. Mathematical Formulas Reference

### 10.1 Core Definitions

**Vector quantization map:**
```
Q: ℝ^d → {0,1}^B,    B = b·d bits total
```

**MSE distortion:**
```
D_mse := E_Q[‖x - Q⁻¹(Q(x))‖₂²]
```

**Inner product distortion:**
```
D_prod := E_Q[|⟨y,x⟩ - ⟨y, Q⁻¹(Q(x))⟩|²]
```

**Unbiasedness requirement:**
```
E_Q[⟨y, Q⁻¹(Q(x))⟩] = ⟨y, x⟩
```

### 10.2 Coordinate Distribution on Unit Sphere

After random rotation, coordinate `y_j` follows:
```
f_X(x) = [Γ(d/2) / (√π · Γ((d-1)/2))] · (1 - x²)^{(d-3)/2},    x ∈ [-1, 1]
```

High-dimension limit:
```
f_X(·) → N(0, 1/d)    as d → ∞
```

### 10.3 Optimal Scalar Quantization

```
C(f_X, b) := min_{c₁ ≤ ... ≤ c_{2^b}}  Σ_{i=1}^{2^b}  ∫_{(c_{i-1}+c_i)/2}^{(c_i+c_{i+1})/2}  |x - c_i|² · f_X(x) dx
```

### 10.4 MSE TurboQuant

```
Quant_mse(x):     y ← Π·x;   idx_j ← nearest(y_j, codebook)
DeQuant_mse(idx):  ỹ_j ← c_{idx_j};   x̃ ← Πᵀ·ỹ
```

MSE bound (Theorem 1):
```
D_mse ≤ (√3 · π / 2) · (1/4^b)    ≈ 2.72 / 4^b
```

### 10.5 QJL Transform

```
Q_qjl(x)   := sign(S · x),          S_{ij} ~ N(0,1)
Q_qjl⁻¹(z) := (√(π/2) / d) · Sᵀ · z
```

Unbiased: `E[⟨y, Q_qjl⁻¹(Q_qjl(x))⟩] = ⟨y, x⟩`

Variance: `Var ≤ (π/(2d)) · ‖y‖₂² · ‖x‖₂²`

### 10.6 Inner-Product TurboQuant (Two-Stage)

```
Quant_prod(x):
    idx ← Quant_mse(x)              [using (b-1) bits]
    r ← x - DeQuant_mse(idx)        [residual]
    qjl ← sign(S · r)               [1 bit per coord]
    γ ← ‖r‖₂
    return (idx, qjl, γ)

DeQuant_prod(idx, qjl, γ):
    x̃_mse ← DeQuant_mse(idx)
    x̃_qjl ← (√(π/2) / d) · γ · Sᵀ · qjl
    return x̃_mse + x̃_qjl
```

**Inner product estimator:**
```
⟨y, x̃⟩ = ⟨y, x̃_mse⟩ + (√(π/2) / d) · γ · ⟨S·y, qjl⟩
```

**Unbiased:** `E[⟨y, x̃⟩] = ⟨y, x⟩`

**Distortion bound (Theorem 2):**
```
D_prod ≤ (√3 · π² · ‖y‖₂² / d) · (1/4^b)
```

Specific values for `D_prod · d / ‖y‖₂²`:

| b | Bound |
|---|---|
| 1 | 1.57 |
| 2 | 0.56 |
| 3 | 0.18 |
| 4 | 0.047 |

### 10.7 Information-Theoretic Lower Bounds (Theorem 3)

```
D_mse(Q) ≥ 1/4^b          (for any randomized quantizer Q)
D_prod(Q) ≥ (1/d) · 1/4^b
```

TurboQuant is within a constant factor ≈ 2.7× of optimal for MSE, and at b=1 within ≈1.45× of optimal.

### 10.8 Shannon Lower Bound

```
D(B) ≥ 2^{-2B/d}
```

where `B` is total bit budget and `d` is dimension.

---

## 11. Hyperparameters

### 11.1 Primary Hyperparameters

| Parameter | Symbol | Typical Values | Notes |
|---|---|---|---|
| Bit-width | b | 2, 3, 4 | Total bits per coordinate |
| MSE bits | b-1 | 1, 2, 3 | Bits for scalar quantization stage |
| QJL bits | 1 | 1 (fixed) | Always 1 bit for sign quantization |
| Head dimension | d | 64, 128 | From model architecture |
| Gamma precision | — | fp16 | Residual norm storage precision |

### 11.2 Key Findings from Experiments

- **3.5 bits/coordinate:** Quality-neutral (score 50.06 vs 50.06 full cache on LongBench)
- **2.5 bits/coordinate:** Marginal quality degradation
- **4 bits/coordinate:** Up to 8x speedup over fp32 on H100

### 11.3 What "3.5 bits" Means

The paper uses non-integer bit-widths by mixing bit allocations:
- Some coordinates get `b` bits, others get `b-1` bits
- Example: 3.5 bits = half coordinates at 4 bits, half at 3 bits
- Alternatively: outlier channels at higher precision, non-outlier at lower

### 11.4 Codebook Parameters

- **Number of centroids:** `2^{b-1}` (fixed by bit-width choice)
- **Distribution used:** Beta-like for exact computation, or Gaussian approximation for large `d`
- **Precomputed:** Yes, solve once per `(d, b)` pair
- **Storage:** Negligible (at most 16 centroids in fp32 = 64 bytes per codebook)

### 11.5 Rotation Matrix

- No tuning needed — any random orthogonal matrix works
- Generated once from QR decomposition of Gaussian matrix
- Fixed seed for reproducibility
- Shared across layers and heads

### 11.6 QJL Projection Matrix

- No tuning needed — any Gaussian random matrix works
- Generated from fixed seed
- Shared across layers and heads

### 11.7 PolarQuant-Specific (if using polar formulation)

| Parameter | Value | Notes |
|---|---|---|
| Recursion depth L | 4 | Paper recommends 4 vs full log₂(d) |
| Level 1 bits | 4 | Wider range [0, 2π) needs more bits |
| Level 2-4 bits | 2 | Concentrated around π/4, fewer bits suffice |
| Codebook mode | offline/online | Offline: precomputed, faster. Online: per-prompt, slightly better quality |

---

## 12. Performance Characteristics

### 12.1 Memory Savings

| Bit-width | Compression vs fp16 | Memory per coordinate |
|---|---|---|
| 2 bits | 8× | 2 bits + amortized overhead |
| 3 bits | ~5.3× | 3 bits + amortized overhead |
| 3.5 bits | ~4.6× | 3.5 bits + amortized overhead |
| 4 bits | 4× | 4 bits + amortized overhead |

### 12.2 Accuracy (Needle-in-Haystack)

| Method | Compression | Score |
|---|---|---|
| Full cache | 1× | 0.997 |
| TurboQuant | 4× | 0.997 |
| PolarQuant | 4× | 0.995 |
| KIVI | 4× | 0.981 |
| SnapKV | 4× | 0.858 |

### 12.3 Accuracy (LongBench Average)

| Configuration | Score |
|---|---|
| Full cache | 50.06 |
| TurboQuant 3.5 bits | 50.06 |
| TurboQuant 2.5 bits | ~marginal degradation |

### 12.4 Latency

**Compression cost per vector:** `O(d²)` — dominated by two matrix-vector multiplies (rotation and QJL projection). For `d = 128`, this is ~16K multiply-adds = negligible on GPU.

**Decompression cost per vector:** `O(d²)` — same order, dominated by inverse rotation and QJL reconstruction.

**Attention speedup:** 4-bit TurboQuant achieves up to **8× speedup** over 32-bit unquantized keys on H100, because attention is memory-bandwidth-bound and reading 4-bit values is 8× faster than 32-bit.

**Nearest-neighbor indexing time:**
```
TurboQuant: 0.0007 - 0.0021 seconds
RabitQ:     597 - 3957 seconds
```

(TurboQuant requires zero indexing time since it's data-oblivious.)

### 12.5 Models Tested

- Gemma (various sizes)
- Mistral

---

## 13. Implementation Considerations

### 13.1 Structured Rotation Alternative

The paper generates `Π` via QR decomposition of a Gaussian matrix. For faster computation, consider using a **randomized Hadamard transform** instead:

```
Π = H · D
```

where `H` is the Walsh-Hadamard matrix and `D` is a random diagonal sign matrix (`D_{ii} ∈ {-1, +1}`). This gives `O(d log d)` rotation instead of `O(d²)`, while maintaining the same distributional properties in high dimensions.

### 13.2 Avoiding Full Matrix Storage for S

The QJL projection matrix `S` is `d × d` with Gaussian entries. Options:
1. **Store explicitly:** 32 KB per head in fp16. Simple, fast.
2. **Seed-based generation:** Generate on-the-fly from a seed using a fast PRNG. Saves memory but adds compute.
3. **Structured random matrix:** Use a randomized Hadamard (like rotation) for `O(d log d)` multiply.

For typical head dimensions (64-128), explicit storage is recommended.

### 13.3 Numerical Precision

- **Rotation and projection:** fp16 is sufficient for `Π` and `S`
- **Codebook centroids:** fp16 or fp32 (only 2^{b-1} values, negligible storage)
- **Gamma:** fp16 (one scalar per vector)
- **Intermediate computations:** fp16 with fp32 accumulation for dot products

### 13.4 Batch Processing

Compress/decompress entire token sequences in parallel:

```python
# Compress batch of KV vectors
Y = X @ Pi.T           # (seq_len, d) @ (d, d) → GEMM
indices = batch_quantize(Y, codebook)
Y_hat = codebook.centroids[indices]
X_mse = Y_hat @ Pi     # (seq_len, d) @ (d, d) → GEMM
R = X - X_mse
gamma = torch.norm(R, dim=-1)  # (seq_len,)
QJL = torch.sign(R @ S.T)      # (seq_len, d) → GEMM + sign
```

### 13.5 CUDA Kernel Priorities

1. **Fused K-attention kernel:** Unpack + decompress + dot product for attention logits
2. **Fused V-attention kernel:** Unpack + decompress + weighted sum for attention output
3. **Compress kernel:** Rotate + quantize + residual + QJL (prefill path)

### 13.6 Memory Layout for GPU

```
// Recommended memory layout (contiguous per head, interleaved across sequence)

// Option A: Separate arrays (simpler, good for initial implementation)
mse_indices[layer][head][seq_pos][packed_bytes]    // (b-1) bits per coord
qjl_signs[layer][head][seq_pos][packed_bytes]      // 1 bit per coord
gamma[layer][head][seq_pos]                         // fp16

// Option B: Interleaved per position (better cache locality for decoding)
compressed[layer][head][seq_pos] = {
    mse_packed[ceil(d*(b-1)/8)],
    qjl_packed[ceil(d/8)],
    gamma_fp16
}
```

### 13.7 Integration Checklist

```
□ Precompute rotation matrix Π (QR of Gaussian, or randomized Hadamard)
□ Precompute QJL projection matrix S (Gaussian, or structured)
□ Precompute Lloyd-Max codebook for target bit-width and head dimension
□ Implement bit-packing for MSE indices and QJL signs
□ Implement compression function (rotate → quantize → residual → QJL)
□ Implement decompression function (lookup → inverse rotate → QJL reconstruct)
□ Implement fused attention kernel (decompress + matmul)
□ Hook into model's KV cache storage (compress after projection, decompress before attention)
□ Validate: check unbiasedness of inner products empirically
□ Benchmark: measure memory savings and latency impact
```

---

## Appendix A: Lloyd-Max Algorithm for Codebook Construction

```python
import numpy as np
from scipy.special import gamma as gamma_fn
from scipy.integrate import quad

def coordinate_pdf(x, d):
    """PDF of a single coordinate of a uniform point on S^{d-1}."""
    if abs(x) >= 1:
        return 0.0
    coeff = gamma_fn(d / 2) / (np.sqrt(np.pi) * gamma_fn((d - 1) / 2))
    return coeff * (1 - x**2) ** ((d - 3) / 2)

def lloyd_max(d, num_bits, max_iter=1000, tol=1e-10):
    """Compute Lloyd-Max codebook for coordinate distribution on S^{d-1}."""
    num_levels = 2 ** num_bits

    # Initialize centroids uniformly in the support
    sigma = 1.0 / np.sqrt(d)
    centroids = np.linspace(-3*sigma, 3*sigma, num_levels)

    for iteration in range(max_iter):
        # Update boundaries (midpoints)
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        boundaries = np.concatenate([[-1.0], boundaries, [1.0]])

        # Update centroids
        new_centroids = np.zeros(num_levels)
        for i in range(num_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            num, _ = quad(lambda x: x * coordinate_pdf(x, d), lo, hi)
            den, _ = quad(lambda x: coordinate_pdf(x, d), lo, hi)
            new_centroids[i] = num / den if den > 1e-15 else centroids[i]

        if np.max(np.abs(new_centroids - centroids)) < tol:
            break
        centroids = new_centroids

    # Final boundaries
    boundaries = (centroids[:-1] + centroids[1:]) / 2

    return centroids, boundaries

# Example: precompute for d=128, b-1=2 bits (4 centroids)
centroids, boundaries = lloyd_max(d=128, num_bits=2)
print(f"Centroids: {centroids}")
print(f"Boundaries: {boundaries}")
```

## Appendix B: Quick Reference — Compression at 3 Bits

For the common case of 3-bit TurboQuant with `d = 128`:

```
MSE stage:  2 bits per coordinate → 4 centroids
QJL stage:  1 bit per coordinate → sign bits
Gamma:      1 × fp16 per vector

Storage per vector:
  MSE:   2 × 128 = 256 bits = 32 bytes
  QJL:   1 × 128 = 128 bits = 16 bytes
  Gamma: 16 bits = 2 bytes
  Total: 400 bits = 50 bytes per vector

vs. fp16 uncompressed: 128 × 16 = 2048 bits = 256 bytes
Compression ratio: 256 / 50 = 5.12×
Effective bits per coordinate: 400 / 128 = 3.125
```

## Appendix C: Key Differences from Related Work

| Method | Approach | Data-dependent? | Needs normalization? | Unbiased IP? |
|---|---|---|---|---|
| **TurboQuant** | Rotation + scalar quant + QJL | No | No | Yes |
| **KIVI** | Per-channel INT quantization | Yes (scale/zp) | Yes | No |
| **PolarQuant** | Rotation + polar + angle quant | No | No (norm stored) | No |
| **QJL** | Random projection + sign | No | No | Yes |
| **Product Quantization** | Subspace codebooks | Yes (training) | Yes | No |

TurboQuant uniquely combines data-oblivious operation, no normalization overhead, AND unbiased inner products.
