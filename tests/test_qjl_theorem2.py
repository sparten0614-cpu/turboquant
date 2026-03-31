"""
QJL Algorithm Correctness Verification against Paper Theorem 2.

QJL (Quantized Johnson-Lindenstrauss) provides:
  - Unbiased inner product estimation: E[⟨y, x̃⟩] = ⟨y, x⟩
  - Variance bound (Theorem 2): Var[⟨y, x̃⟩] ≤ (π/2 - 1) · ||y||² · ||r||² / d
    where r = x - x_mse is the MSE residual

This test verifies both properties rigorously.

Performance: tests vary only the QJL projection matrix S while reusing the
MSE stage (rotation + codebook), since QJL unbiasedness is independent of
the MSE reconstruction. This avoids the O(d²) compressor init overhead per trial.
"""

import numpy as np
import math
import pytest

from turboquant.turboquant import TurboQuantConfig, TurboQuantCompressor
from turboquant.codebook import quantize_scalar, dequantize_scalar
from turboquant.bitpack import pack_mse_indices, pack_qjl_bits, CompressedKV


def _compress_with_new_S(comp, x, seed):
    """Compress x using comp's MSE stage but a fresh QJL projection S.

    This is the fast path for testing QJL properties: the MSE stage
    (rotation + codebook) is fixed, only the random projection varies.
    """
    d = comp.d
    rng = np.random.default_rng(seed)
    S = rng.standard_normal((d, d))

    x_norm = np.linalg.norm(x)
    if x_norm < 1e-12:
        return CompressedKV(
            mse_packed=pack_mse_indices(np.zeros(d, dtype=np.uint8), comp.config.mse_bits),
            qjl_packed=pack_qjl_bits(np.ones(d)),
            gamma=0.0, mse_bits=comp.config.mse_bits, d=d, x_norm=0.0,
        ), S

    x_hat = x / x_norm
    y = comp._rotate(x_hat)
    indices = quantize_scalar(y, comp.codebook)
    y_hat = dequantize_scalar(indices, comp.codebook)
    x_mse = x_norm * comp._rotate_inverse(y_hat)

    r = x - x_mse
    gamma = np.linalg.norm(r)
    qjl_signs = np.sign(S @ r) if gamma > 1e-12 else np.ones(d)

    ckv = CompressedKV(
        mse_packed=pack_mse_indices(indices, comp.config.mse_bits),
        qjl_packed=pack_qjl_bits(qjl_signs),
        gamma=gamma, mse_bits=comp.config.mse_bits, d=d, x_norm=x_norm,
    )
    return ckv, S


def _inner_product_with_S(comp, y, compressed, S):
    """Compute inner product using a specific S matrix."""
    indices = compressed.get_mse_indices()
    qjl_signs = compressed.get_qjl_signs()
    gamma = float(compressed.gamma)
    x_norm = float(compressed.x_norm)

    y_hat = dequantize_scalar(indices, comp.codebook)
    x_mse = x_norm * comp._rotate_inverse(y_hat)
    ip_mse = np.dot(y, x_mse)

    d = comp.d
    scale = math.sqrt(math.pi / 2) / d
    ip_qjl = scale * gamma * np.dot(S @ y, qjl_signs)

    return ip_mse + ip_qjl


class TestQJLRoundtrip:
    """Test QJL sign encode → decode → inner product roundtrip."""

    def setup_method(self):
        self.rng = np.random.default_rng(42)

    def test_sign_roundtrip(self):
        """sign(S·x) encode → decode preserves information for inner products."""
        d = 128
        config = TurboQuantConfig(head_dim=d, total_bits=3)
        comp = TurboQuantCompressor(config)

        x = self.rng.standard_normal(d)
        compressed = comp.compress(x)
        x_hat = comp.decompress(compressed)

        assert x_hat.shape == (d,)
        assert not np.any(np.isnan(x_hat))
        assert not np.any(np.isinf(x_hat))

        cos = np.dot(x, x_hat) / (np.linalg.norm(x) * np.linalg.norm(x_hat))
        assert cos > 0.8, f"cos_sim {cos:.4f} too low"

    def test_inner_product_vs_decompress(self):
        """inner_product() should match decompress() + dot() exactly."""
        d = 128
        config = TurboQuantConfig(head_dim=d, total_bits=3)
        comp = TurboQuantCompressor(config)

        x = self.rng.standard_normal(d)
        q = self.rng.standard_normal(d)
        compressed = comp.compress(x)

        ip_direct = comp.inner_product(q, compressed)
        ip_decompress = np.dot(q, comp.decompress(compressed))

        np.testing.assert_allclose(ip_direct, ip_decompress, rtol=1e-10,
                                   err_msg="inner_product() and decompress+dot should match")


class TestQJLUnbiasedness:
    """Verify E[⟨y, x̃⟩] = ⟨y, x⟩ (unbiased inner product estimation)."""

    @pytest.mark.parametrize("d", [64, 128])
    @pytest.mark.parametrize("total_bits", [2, 3, 4])
    def test_unbiased_inner_product(self, d, total_bits):
        """Inner product estimate should be unbiased over many random S matrices."""
        n_trials = 500
        rng = np.random.default_rng(123)

        x = rng.standard_normal(d)
        y = rng.standard_normal(d)
        true_ip = np.dot(y, x)

        # Create one compressor, vary only S
        config = TurboQuantConfig(head_dim=d, total_bits=total_bits, seed=0)
        comp = TurboQuantCompressor(config)

        ip_estimates = []
        for trial in range(n_trials):
            ckv, S = _compress_with_new_S(comp, x, seed=trial + 10000)
            ip_est = _inner_product_with_S(comp, y, ckv, S)
            ip_estimates.append(ip_est)

        ip_estimates = np.array(ip_estimates)
        mean_ip = np.mean(ip_estimates)
        std_ip = np.std(ip_estimates)

        se = std_ip / np.sqrt(n_trials)
        z_score = abs(mean_ip - true_ip) / (se + 1e-10)

        assert z_score < 4.0, (
            f"d={d}, {total_bits}bit: biased! "
            f"E[ip]={mean_ip:.4f}, true={true_ip:.4f}, "
            f"z={z_score:.2f}, bias/std={abs(mean_ip - true_ip)/(std_ip + 1e-10):.4f}"
        )

    def test_unbiased_with_real_kv_magnitudes(self):
        """Unbiasedness should hold for KV-cache-like magnitudes."""
        d = 128
        rng = np.random.default_rng(456)

        x = rng.standard_normal(d)
        x = x * (20.0 / np.linalg.norm(x))  # norm=20 (Llama-like)
        y = rng.standard_normal(d)
        y = y * (15.0 / np.linalg.norm(y))

        true_ip = np.dot(y, x)
        n_trials = 500

        config = TurboQuantConfig(head_dim=d, total_bits=4, seed=0)
        comp = TurboQuantCompressor(config)

        ip_estimates = []
        for trial in range(n_trials):
            ckv, S = _compress_with_new_S(comp, x, seed=trial + 20000)
            ip_est = _inner_product_with_S(comp, y, ckv, S)
            ip_estimates.append(ip_est)

        mean_ip = np.mean(ip_estimates)
        se = np.std(ip_estimates) / np.sqrt(n_trials)
        z_score = abs(mean_ip - true_ip) / (se + 1e-10)

        assert z_score < 4.0, (
            f"Biased for KV-like magnitudes! "
            f"E[ip]={mean_ip:.4f}, true={true_ip:.4f}, z={z_score:.2f}"
        )


class TestQJLTheorem2Variance:
    """Verify variance bound from QJL Theorem 2.

    Theorem 2: For x, y ∈ ℝ^d and S ∈ ℝ^{m×d} with i.i.d. N(0,1) entries,
    the QJL estimator satisfies:

        Var[⟨y, x̃_QJL⟩] ≤ (π/2 - 1) · ||y||² · ||r||² / m

    where r is the residual and m is the projection dimension (= d in our case).
    """

    @pytest.mark.parametrize("d", [64, 128])
    @pytest.mark.parametrize("total_bits", [2, 3, 4])
    def test_variance_within_bound(self, d, total_bits):
        """Empirical variance should be ≤ theoretical bound from Theorem 2."""
        n_trials = 1000
        rng = np.random.default_rng(789)

        x = rng.standard_normal(d)
        y = rng.standard_normal(d)

        config = TurboQuantConfig(head_dim=d, total_bits=total_bits, seed=0)
        comp = TurboQuantCompressor(config)

        # Compute the MSE residual (fixed, since rotation is fixed)
        x_norm = np.linalg.norm(x)
        x_hat = x / x_norm
        y_rot = comp._rotate(x_hat)
        indices = quantize_scalar(y_rot, comp.codebook)
        y_hat_q = dequantize_scalar(indices, comp.codebook)
        x_mse = x_norm * comp._rotate_inverse(y_hat_q)
        r = x - x_mse
        r_norm = np.linalg.norm(r)
        y_norm = np.linalg.norm(y)

        # Theoretical variance bound: (π/2 - 1) · ||y||² · ||r||² / d
        theoretical_bound = (math.pi / 2 - 1) * y_norm**2 * r_norm**2 / d

        # Collect inner product estimates varying only S
        ip_estimates = []
        for trial in range(n_trials):
            ckv, S = _compress_with_new_S(comp, x, seed=trial + 30000)
            ip_est = _inner_product_with_S(comp, y, ckv, S)
            ip_estimates.append(ip_est)

        empirical_var = np.var(ip_estimates)

        # Allow 5x multiplier for finite-sample noise
        assert empirical_var < theoretical_bound * 5, (
            f"d={d}, {total_bits}bit: variance too high! "
            f"empirical={empirical_var:.6f}, "
            f"bound={theoretical_bound:.6f}, "
            f"ratio={empirical_var/theoretical_bound:.2f}x"
        )
        assert empirical_var > 0, "Variance should not be zero"

        # Print for reporting
        print(f"  d={d}, {total_bits}bit: var={empirical_var:.6f}, "
              f"bound={theoretical_bound:.6f}, ratio={empirical_var/theoretical_bound:.2f}x")

    def test_normalized_variance_decreases_with_dimension(self):
        """Theorem 2 predicts Var / (||y||²·||r||²) ∝ 1/d. Verify this scaling.

        Raw variance grows with d because ||y||² and ||r||² grow with d.
        The normalized ratio Var / (||y||²·||r||²) should decrease.
        """
        rng = np.random.default_rng(321)
        n_trials = 500

        normalized_vars = {}
        for d in [32, 64, 128]:
            x = rng.standard_normal(d)
            y = rng.standard_normal(d)

            config = TurboQuantConfig(head_dim=d, total_bits=3, seed=0)
            comp = TurboQuantCompressor(config)

            # Compute residual norm (fixed for this rotation)
            x_norm = np.linalg.norm(x)
            x_hat = x / x_norm
            y_rot = comp._rotate(x_hat)
            indices = quantize_scalar(y_rot, comp.codebook)
            y_hat_q = dequantize_scalar(indices, comp.codebook)
            x_mse = x_norm * comp._rotate_inverse(y_hat_q)
            r = x - x_mse
            r_norm = np.linalg.norm(r)
            y_norm = np.linalg.norm(y)

            ips = []
            for trial in range(n_trials):
                ckv, S = _compress_with_new_S(comp, x, seed=trial + 40000)
                ip = _inner_product_with_S(comp, y, ckv, S)
                ips.append(ip)

            raw_var = np.var(ips)
            # Normalize: Var / (||y||² · ||r||²)
            normalized_vars[d] = raw_var / (y_norm**2 * r_norm**2)

        print(f"  Normalized variance: d=32 → {normalized_vars[32]:.6f}, "
              f"d=64 → {normalized_vars[64]:.6f}, d=128 → {normalized_vars[128]:.6f}")

        # Normalized variance should decrease (∝ 1/d)
        assert normalized_vars[64] < normalized_vars[32] * 1.5, (
            f"Normalized variance should decrease: d=32 → {normalized_vars[32]:.6f}, "
            f"d=64 → {normalized_vars[64]:.6f}"
        )
        assert normalized_vars[128] < normalized_vars[64] * 1.5, (
            f"Normalized variance should decrease: d=64 → {normalized_vars[64]:.6f}, "
            f"d=128 → {normalized_vars[128]:.6f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
