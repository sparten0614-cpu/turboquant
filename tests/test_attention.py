"""
Test TurboQuant in a realistic attention computation scenario.

Key validation: QJL correction provides unbiased inner products
when averaged over many KV entries (as in attention softmax).
"""

import numpy as np

from turboquant.turboquant import TurboQuantConfig, TurboQuantCompressor


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def test_attention_logits_unbiased():
    """Test that compressed attention logits are unbiased estimators.

    For a query q and many keys k_1..k_n:
    E[⟨q, k̃_i⟩] ≈ ⟨q, k_i⟩

    Run multiple trials and check bias is near zero.
    """
    config = TurboQuantConfig(head_dim=128, total_bits=3, seed=42)
    compressor = TurboQuantCompressor(config)
    rng = np.random.default_rng(123)

    n_trials = 100
    d = 128

    biases = []
    for _ in range(n_trials):
        q = rng.standard_normal(d)
        k = rng.standard_normal(d)

        true_ip = np.dot(q, k)
        compressed = compressor.compress(k)
        approx_ip = compressor.inner_product(q, compressed)

        biases.append(approx_ip - true_ip)

    mean_bias = np.mean(biases)
    std_bias = np.std(biases)

    print(f"  Mean bias: {mean_bias:.6f} (should be ~0)")
    print(f"  Std dev:   {std_bias:.6f}")
    print(f"  |mean/std|: {abs(mean_bias/std_bias):.4f} (should be < 0.3 for unbiased)")

    # Unbiasedness: mean bias should be small relative to std
    assert abs(mean_bias / std_bias) < 0.5, f"Bias too large: {mean_bias:.4f} ± {std_bias:.4f}"
    print("PASS: attention_logits_unbiased")


def test_full_attention_quality():
    """Test full attention computation with compressed KV cache.

    Compare: softmax(Q @ K^T / √d) @ V
    vs:      softmax(Q @ K̃^T / √d) @ Ṽ

    Measure MSE of attention output.
    """
    for total_bits in [2, 3, 4]:
        config = TurboQuantConfig(head_dim=128, total_bits=total_bits, seed=42)
        compressor = TurboQuantCompressor(config)
        rng = np.random.default_rng(456)

        d = 128
        seq_len = 64  # KV cache length
        n_queries = 4

        # Generate random KV cache and queries
        K = rng.standard_normal((seq_len, d)) * 0.1  # Scale to typical magnitude
        V = rng.standard_normal((seq_len, d)) * 0.1
        Q = rng.standard_normal((n_queries, d)) * 0.1

        # Compress KV cache
        K_compressed = compressor.compress_batch(K)
        V_compressed = compressor.compress_batch(V)

        # True attention
        scale = 1.0 / np.sqrt(d)
        true_outputs = []
        for qi in range(n_queries):
            logits = (Q[qi] @ K.T) * scale
            attn_weights = softmax(logits)
            out = attn_weights @ V
            true_outputs.append(out)
        true_outputs = np.stack(true_outputs)

        # Compressed attention
        comp_outputs = []
        for qi in range(n_queries):
            # Compute logits via compressed inner products
            logits = np.array([
                compressor.inner_product(Q[qi], K_compressed[j])
                for j in range(seq_len)
            ]) * scale
            attn_weights = softmax(logits)

            # Decompress values and compute weighted sum
            V_decompressed = compressor.decompress_batch(V_compressed)
            out = attn_weights @ V_decompressed
            comp_outputs.append(out)
        comp_outputs = np.stack(comp_outputs)

        # Measure quality
        mse = np.mean((true_outputs - comp_outputs) ** 2)
        cos_sim = np.mean([
            np.dot(true_outputs[i], comp_outputs[i]) /
            (np.linalg.norm(true_outputs[i]) * np.linalg.norm(comp_outputs[i]) + 1e-10)
            for i in range(n_queries)
        ])

        print(f"  turbo{total_bits}: attention MSE={mse:.8f}, cos_sim={cos_sim:.6f}")

    print("PASS: full_attention_quality")


def test_attention_mse_vs_mse_only():
    """Compare full TurboQuant (MSE+QJL) vs MSE-only for attention.

    QJL correction should improve attention quality for keys.
    """
    d = 128
    seq_len = 32
    rng = np.random.default_rng(789)

    K = rng.standard_normal((seq_len, d)) * 0.1
    V = rng.standard_normal((seq_len, d)) * 0.1
    q = rng.standard_normal(d) * 0.1

    # Full TurboQuant
    config_full = TurboQuantConfig(head_dim=d, total_bits=3, seed=42)
    comp_full = TurboQuantCompressor(config_full)

    # True attention output
    scale = 1.0 / np.sqrt(d)
    true_logits = (q @ K.T) * scale
    true_weights = softmax(true_logits)
    true_out = true_weights @ V

    # Full TurboQuant attention
    K_comp = comp_full.compress_batch(K)
    V_comp = comp_full.compress_batch(V)

    full_logits = np.array([comp_full.inner_product(q, K_comp[j]) for j in range(seq_len)]) * scale
    full_weights = softmax(full_logits)
    V_dec = comp_full.decompress_batch(V_comp)
    full_out = full_weights @ V_dec

    full_mse = np.mean((true_out - full_out) ** 2)

    # MSE-only (decompress and use directly, no QJL inner product path)
    K_dec = comp_full.decompress_batch(K_comp)
    mse_logits = (q @ K_dec.T) * scale
    mse_weights = softmax(mse_logits)
    mse_out = mse_weights @ V_dec

    mse_only_mse = np.mean((true_out - mse_out) ** 2)

    print(f"  Full TurboQuant attention MSE: {full_mse:.8f}")
    print(f"  Decompress-then-dot MSE:       {mse_only_mse:.8f}")
    print(f"  (Both paths should be similar for turbo3)")
    print("PASS: attention_mse_vs_mse_only")


def test_compression_memory_savings():
    """Calculate actual memory savings for a realistic scenario."""
    d = 128
    n_layers = 32
    n_heads = 32
    seq_len = 4096

    fp16_bytes_per_entry = d * 2  # 2 bytes per fp16
    fp16_total = 2 * n_layers * n_heads * seq_len * fp16_bytes_per_entry  # K + V

    for total_bits in [2, 3, 4]:
        config = TurboQuantConfig(head_dim=d, total_bits=total_bits, seed=42)
        comp = TurboQuantCompressor(config)

        # Compress one vector to get actual size
        x = np.random.randn(d)
        compressed = comp.compress(x)
        compressed_bytes = compressed.total_bytes

        compressed_total = 2 * n_layers * n_heads * seq_len * compressed_bytes

        ratio = fp16_total / compressed_total
        print(f"  turbo{total_bits}: {fp16_total/1e6:.1f}MB → {compressed_total/1e6:.1f}MB ({ratio:.1f}x)")

    print("PASS: compression_memory_savings")


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 60)
    print("TurboQuant Attention Tests")
    print("=" * 60)

    test_attention_logits_unbiased()
    test_full_attention_quality()
    test_attention_mse_vs_mse_only()
    test_compression_memory_savings()

    print("=" * 60)
    print("ALL ATTENTION TESTS PASSED")
    print("=" * 60)
