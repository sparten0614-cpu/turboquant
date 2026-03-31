"""
Lloyd-Max optimal scalar quantizer for TurboQuant.

The codebook is precomputed for the coordinate distribution after random rotation.
For vectors on the unit sphere S^{d-1}, after rotation each coordinate follows:
    f(x) ∝ (1 - x²)^{(d-3)/2}   on [-1, 1]

For large d (d >= 64), this is well-approximated by N(0, 1/d).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy import stats, integrate
from scipy.special import gamma as gamma_func


@dataclass
class Codebook:
    """Lloyd-Max codebook for scalar quantization."""
    centroids: np.ndarray    # Shape (num_levels,), sorted
    boundaries: np.ndarray   # Shape (num_levels - 1,), decision boundaries
    bits: int                # Number of bits (log2 of num_levels)
    dim: int                 # Head dimension used to compute distribution

    @property
    def num_levels(self) -> int:
        return len(self.centroids)


def beta_pdf(x: float, d: int) -> float:
    """PDF of a coordinate on the unit sphere after random rotation.

    f(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^{(d-3)/2}
    Supported on [-1, 1].
    """
    if abs(x) >= 1.0:
        return 0.0
    coeff = gamma_func(d / 2) / (np.sqrt(np.pi) * gamma_func((d - 1) / 2))
    return coeff * (1 - x**2) ** ((d - 3) / 2)


def gaussian_pdf(x: float, d: int) -> float:
    """Gaussian approximation N(0, 1/d) for the coordinate distribution.

    Good approximation for d >= 64.
    """
    sigma = 1.0 / np.sqrt(d)
    return stats.norm.pdf(x, loc=0, scale=sigma)


def lloyd_max(d: int, num_bits: int, use_gaussian: bool = True,
              max_iter: int = 200, tol: float = 1e-10) -> Codebook:
    """Compute optimal Lloyd-Max codebook for the coordinate distribution.

    Args:
        d: Head dimension (determines the distribution)
        num_bits: Bits per coordinate for this stage
        use_gaussian: If True, use N(0, 1/d) approximation (recommended for d >= 64)
        max_iter: Maximum Lloyd-Max iterations
        tol: Convergence tolerance

    Returns:
        Codebook with optimal centroids and decision boundaries
    """
    num_levels = 2 ** num_bits
    sigma = 1.0 / np.sqrt(d)

    if use_gaussian:
        pdf = lambda x: stats.norm.pdf(x, 0, sigma)
        cdf = lambda x: stats.norm.cdf(x, 0, sigma)
        support = (-4 * sigma, 4 * sigma)  # Effectively [-4/√d, 4/√d]
    else:
        pdf = lambda x: beta_pdf(x, d)
        support = (-1.0, 1.0)

    # Initialize centroids uniformly in the support
    centroids = np.linspace(support[0], support[1], num_levels + 2)[1:-1]

    for iteration in range(max_iter):
        old_centroids = centroids.copy()

        # Update boundaries: midpoints between adjacent centroids
        boundaries = (centroids[:-1] + centroids[1:]) / 2

        # Update centroids: conditional expectation within each region
        # Region i: [b_{i-1}, b_i] where b_0 = -inf, b_{num_levels} = +inf
        extended_bounds = [support[0] - 1] + list(boundaries) + [support[1] + 1]

        for i in range(num_levels):
            lo = max(extended_bounds[i], support[0])
            hi = min(extended_bounds[i + 1], support[1])

            if lo >= hi:
                continue

            # Numerator: ∫ x · f(x) dx over [lo, hi]
            num, _ = integrate.quad(lambda x: x * pdf(x), lo, hi)
            # Denominator: ∫ f(x) dx over [lo, hi]
            den, _ = integrate.quad(pdf, lo, hi)

            if den > 1e-15:
                centroids[i] = num / den

        # Check convergence
        if np.max(np.abs(centroids - old_centroids)) < tol:
            break

    # Final boundaries
    boundaries = (centroids[:-1] + centroids[1:]) / 2

    return Codebook(
        centroids=centroids,
        boundaries=boundaries,
        bits=num_bits,
        dim=d,
    )


def precompute_codebooks(d: int, bit_widths: list[int] = [1, 2, 3]) -> dict[int, Codebook]:
    """Precompute codebooks for all needed bit-widths.

    For TurboQuant with total b bits: MSE stage uses (b-1) bits.
    - turbo2 (b=2): MSE uses 1 bit → 2 centroids
    - turbo3 (b=3): MSE uses 2 bits → 4 centroids
    - turbo4 (b=4): MSE uses 3 bits → 8 centroids
    """
    codebooks = {}
    for bits in bit_widths:
        codebooks[bits] = lloyd_max(d, bits)
    return codebooks


def quantize_scalar(values: np.ndarray, codebook: Codebook) -> np.ndarray:
    """Quantize each value to the nearest centroid.

    Args:
        values: Array of scalar values to quantize
        codebook: Lloyd-Max codebook

    Returns:
        indices: Array of codebook indices, same shape as values
    """
    # Use searchsorted on boundaries for O(log n) per value
    indices = np.searchsorted(codebook.boundaries, values)
    return indices.astype(np.uint8)


def dequantize_scalar(indices: np.ndarray, codebook: Codebook) -> np.ndarray:
    """Look up centroid values from indices.

    Args:
        indices: Codebook indices
        codebook: Lloyd-Max codebook

    Returns:
        Reconstructed values
    """
    return codebook.centroids[indices]
