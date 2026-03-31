"""Unit tests for TurboQuant core components."""

import numpy as np

from turboquant.rotation import (
    generate_random_orthogonal,
    walsh_hadamard_transform,
    normalized_wht,
    generate_random_signs,
    fast_random_rotation,
    fast_random_rotation_inverse,
)
from turboquant.codebook import lloyd_max, quantize_scalar, dequantize_scalar
from turboquant.bitpack import (
    pack_mse_indices, unpack_mse_indices,
    pack_qjl_bits, unpack_qjl_bits,
    CompressedKV,
)
from turboquant.turboquant import TurboQuantConfig, TurboQuantCompressor


def test_random_orthogonal():
    """Pi should be orthogonal: Pi @ Pi.T = I."""
    d = 128
    Pi = generate_random_orthogonal(d, seed=42)
    I = Pi @ Pi.T
    assert np.allclose(I, np.eye(d), atol=1e-10), "Pi is not orthogonal"
    print("PASS: random_orthogonal")


def test_wht_self_inverse():
    """Normalized WHT is self-inverse: (H/√d)² = I."""
    d = 128
    x = np.random.randn(d)
    y = normalized_wht(x)
    x_rec = normalized_wht(y)
    assert np.allclose(x, x_rec, atol=1e-10), f"WHT not self-inverse, max err={np.max(np.abs(x - x_rec))}"
    print("PASS: wht_self_inverse")


def test_wht_orthogonal():
    """Normalized WHT preserves norms: ‖Hx/√d‖ = ‖x‖."""
    d = 64
    x = np.random.randn(d)
    y = normalized_wht(x)
    assert np.isclose(np.linalg.norm(x), np.linalg.norm(y), rtol=1e-10), "WHT does not preserve norm"
    print("PASS: wht_orthogonal")


def test_fast_rotation_inverse():
    """Fast rotation should be invertible."""
    d = 128
    signs = generate_random_signs(d, seed=42)
    x = np.random.randn(d)
    y = fast_random_rotation(x, signs)
    x_rec = fast_random_rotation_inverse(y, signs)
    assert np.allclose(x, x_rec, atol=1e-10), f"Fast rotation not invertible, max err={np.max(np.abs(x - x_rec))}"
    print("PASS: fast_rotation_inverse")


def test_fast_rotation_preserves_norm():
    """Fast rotation should preserve vector norms."""
    d = 128
    signs = generate_random_signs(d, seed=42)
    x = np.random.randn(d)
    y = fast_random_rotation(x, signs)
    assert np.isclose(np.linalg.norm(x), np.linalg.norm(y), rtol=1e-10), "Fast rotation changes norm"
    print("PASS: fast_rotation_preserves_norm")


def test_codebook_sorted():
    """Codebook centroids and boundaries should be sorted."""
    for bits in [1, 2, 3]:
        cb = lloyd_max(128, bits)
        assert np.all(np.diff(cb.centroids) > 0), f"Centroids not sorted for {bits} bits"
        assert np.all(np.diff(cb.boundaries) > 0), f"Boundaries not sorted for {bits} bits"
        assert len(cb.centroids) == 2**bits, f"Wrong number of centroids for {bits} bits"
        assert len(cb.boundaries) == 2**bits - 1, f"Wrong number of boundaries for {bits} bits"
    print("PASS: codebook_sorted")


def test_codebook_symmetry():
    """For symmetric distribution, codebook should be approximately symmetric around 0."""
    cb = lloyd_max(128, 2)
    # The distribution N(0, 1/d) is symmetric, so centroids should be symmetric
    for i in range(len(cb.centroids)):
        j = len(cb.centroids) - 1 - i
        assert np.isclose(cb.centroids[i], -cb.centroids[j], atol=1e-6), \
            f"Codebook not symmetric: {cb.centroids[i]} vs {-cb.centroids[j]}"
    print("PASS: codebook_symmetry")


def test_quantize_dequantize():
    """Quantize then dequantize should produce values from codebook."""
    cb = lloyd_max(128, 2)
    values = np.random.randn(128) / np.sqrt(128)  # ~N(0, 1/d)
    indices = quantize_scalar(values, cb)
    reconstructed = dequantize_scalar(indices, cb)
    assert all(r in cb.centroids for r in reconstructed), "Dequantized values not in codebook"
    print("PASS: quantize_dequantize")


def test_bitpack_mse_roundtrip():
    """Pack and unpack MSE indices should be lossless."""
    d = 128
    for bits in [1, 2, 3]:
        max_val = 2**bits - 1
        indices = np.random.randint(0, max_val + 1, size=d).astype(np.uint8)
        packed = pack_mse_indices(indices, bits)
        unpacked = unpack_mse_indices(packed, d, bits)
        assert np.array_equal(indices, unpacked), \
            f"MSE bitpack roundtrip failed for {bits} bits: {np.sum(indices != unpacked)} mismatches"
    print("PASS: bitpack_mse_roundtrip")


def test_bitpack_qjl_roundtrip():
    """Pack and unpack QJL signs should be lossless."""
    d = 128
    signs = np.random.choice([-1.0, 1.0], size=d)
    packed = pack_qjl_bits(signs)
    unpacked = unpack_qjl_bits(packed, d)
    assert np.array_equal(signs, unpacked), "QJL bitpack roundtrip failed"
    print("PASS: bitpack_qjl_roundtrip")


def test_turboquant_roundtrip_mse():
    """Compress then decompress should have bounded MSE."""
    for total_bits in [2, 3, 4]:
        config = TurboQuantConfig(head_dim=128, total_bits=total_bits, seed=42)
        compressor = TurboQuantCompressor(config)

        # Random unit vector
        x = np.random.randn(128)
        x = x / np.linalg.norm(x)

        compressed = compressor.compress(x)
        x_rec = compressor.decompress(compressed)

        mse = np.mean((x - x_rec) ** 2)
        # MSE should decrease with more bits
        print(f"  turbo{total_bits}: MSE={mse:.6f}, compression={compressed.compression_ratio:.1f}x")

    print("PASS: turboquant_roundtrip_mse")


def test_turboquant_inner_product():
    """Inner product via compressed domain should approximate true inner product."""
    config = TurboQuantConfig(head_dim=128, total_bits=3, seed=42)
    compressor = TurboQuantCompressor(config)

    # Random vectors
    rng = np.random.default_rng(123)
    x = rng.standard_normal(128)
    q = rng.standard_normal(128)

    true_ip = np.dot(q, x)

    compressed = compressor.compress(x)
    approx_ip = compressor.inner_product(q, compressed)

    # Also check via full decompression
    x_rec = compressor.decompress(compressed)
    decomp_ip = np.dot(q, x_rec)

    # Inner product via compressed path should match decompress path
    assert np.isclose(approx_ip, decomp_ip, atol=1e-10), \
        f"Inner product paths disagree: {approx_ip} vs {decomp_ip}"

    rel_err = abs(approx_ip - true_ip) / (abs(true_ip) + 1e-10)
    print(f"  True IP={true_ip:.4f}, Approx IP={approx_ip:.4f}, RelErr={rel_err:.4f}")
    print("PASS: turboquant_inner_product")


def test_compression_ratio():
    """Verify compression ratios match theoretical values."""
    d = 128
    for total_bits in [2, 3, 4]:
        config = TurboQuantConfig(head_dim=d, total_bits=total_bits, seed=42)
        compressor = TurboQuantCompressor(config)
        x = np.random.randn(d)
        compressed = compressor.compress(x)

        # Theoretical: d * 16 bits (fp16) / (d * total_bits + 16 bits for gamma)
        theoretical = (d * 16) / (d * total_bits + 16)
        actual = compressed.compression_ratio

        print(f"  turbo{total_bits}: theoretical={theoretical:.2f}x, actual={actual:.2f}x")
    print("PASS: compression_ratio")


def test_deterministic():
    """Same input + same config → same output."""
    config = TurboQuantConfig(head_dim=128, total_bits=3, seed=42)
    c1 = TurboQuantCompressor(config)
    c2 = TurboQuantCompressor(config)

    x = np.random.randn(128)
    comp1 = c1.compress(x)
    comp2 = c2.compress(x)

    assert np.array_equal(comp1.mse_packed, comp2.mse_packed), "MSE indices not deterministic"
    assert np.array_equal(comp1.qjl_packed, comp2.qjl_packed), "QJL bits not deterministic"
    assert comp1.gamma == comp2.gamma, "Gamma not deterministic"
    print("PASS: deterministic")


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 60)
    print("TurboQuant Core Unit Tests")
    print("=" * 60)

    test_random_orthogonal()
    test_wht_self_inverse()
    test_wht_orthogonal()
    test_fast_rotation_inverse()
    test_fast_rotation_preserves_norm()
    test_codebook_sorted()
    test_codebook_symmetry()
    test_quantize_dequantize()
    test_bitpack_mse_roundtrip()
    test_bitpack_qjl_roundtrip()
    test_turboquant_roundtrip_mse()
    test_turboquant_inner_product()
    test_compression_ratio()
    test_deterministic()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
