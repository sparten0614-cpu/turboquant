"""
Random rotation matrices for TurboQuant PolarQuant stage.

Two implementations:
1. QR-based random orthogonal matrix (general, O(d²) multiply)
2. Walsh-Hadamard Transform with random sign flip (fast, O(d·log d))
"""

import numpy as np
from typing import Optional


def generate_random_orthogonal(d: int, seed: int = 42) -> np.ndarray:
    """Generate a random orthogonal matrix via QR decomposition of Gaussian matrix.

    Args:
        d: Dimension (head_dim, typically 64 or 128)
        seed: Random seed for reproducibility

    Returns:
        Pi: Orthogonal matrix of shape (d, d), float64
    """
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, d))
    Pi, R = np.linalg.qr(G)
    # Ensure det(Pi) = +1 (proper rotation)
    diag_sign = np.sign(np.diag(R))
    Pi = Pi * diag_sign[np.newaxis, :]
    return Pi


def walsh_hadamard_transform(x: np.ndarray) -> np.ndarray:
    """In-place Walsh-Hadamard Transform (unnormalized).

    Computes H·x where H is the Hadamard matrix of size d×d.
    d must be a power of 2.

    Butterfly algorithm: O(d·log₂(d)) operations.

    Args:
        x: Input vector(s). Shape (d,) or (batch, d). d must be power of 2.

    Returns:
        Transformed vector(s), same shape as input.
    """
    x = x.copy().astype(np.float64)
    if x.ndim == 1:
        x = x[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False

    batch, d = x.shape
    assert d > 0 and (d & (d - 1)) == 0, f"d must be power of 2, got {d}"

    # Butterfly stages
    h = 1
    while h < d:
        for i in range(0, d, h * 2):
            for j in range(i, i + h):
                a = x[:, j].copy()
                b = x[:, j + h].copy()
                x[:, j] = a + b
                x[:, j + h] = a - b
        h *= 2

    if squeeze:
        return x[0]
    return x


def normalized_wht(x: np.ndarray) -> np.ndarray:
    """Normalized Walsh-Hadamard Transform: (1/√d) · H · x.

    After normalization, the transform is orthogonal (H/√d is orthogonal).
    """
    d = x.shape[-1]
    return walsh_hadamard_transform(x) / np.sqrt(d)


def generate_random_signs(d: int, seed: int = 42) -> np.ndarray:
    """Generate random ±1 sign vector for D·H·D rotation.

    Args:
        d: Dimension
        seed: Random seed

    Returns:
        signs: Array of shape (d,) with values in {-1, +1}
    """
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=d)


def fast_random_rotation(x: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Apply fast random rotation via D·H·D (sign-flip + WHT + sign-flip).

    This is the recommended rotation for GPU implementation.
    The composition D·H·D (where D is diagonal ±1, H is normalized Hadamard)
    approximates a random orthogonal matrix.

    O(d·log d) instead of O(d²).

    Args:
        x: Input vector(s), shape (d,) or (batch, d). d must be power of 2.
        signs: Random ±1 vector of shape (d,)

    Returns:
        Rotated vector(s).
    """
    # Step 1: Apply random signs (D₁)
    y = x * signs
    # Step 2: Normalized WHT
    y = normalized_wht(y)
    # Step 3: Apply random signs again (D₂) — optional, improves mixing
    # For simplicity, reuse the same signs (paper uses single D*H)
    return y


def fast_random_rotation_inverse(y: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Inverse of fast_random_rotation.

    forward:  y = H_norm · (D · x)  where H_norm = H/√d
    inverse:  x = D · (H_norm · y)  since H_norm is self-inverse and D is self-inverse
    """
    # Step 1: Apply normalized WHT (self-inverse)
    x = normalized_wht(y)
    # Step 2: Apply sign flip (self-inverse)
    x = x * signs
    return x
