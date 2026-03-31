"""
TurboQuant: Two-stage KV cache compression.

Stage 1 (PolarQuant/MSE): Random rotation + Lloyd-Max scalar quantization
Stage 2 (QJL): 1-bit sign quantization of residual for inner-product correction

This is the core encode/decode implementation.
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Optional

from .rotation import (
    generate_random_orthogonal,
    generate_random_signs,
    fast_random_rotation,
    fast_random_rotation_inverse,
)
from .codebook import (
    Codebook,
    lloyd_max,
    precompute_codebooks,
    quantize_scalar,
    dequantize_scalar,
)
from .bitpack import CompressedKV, pack_mse_indices, pack_qjl_bits


GOLDEN_RATIO = 0x9E3779B9  # Fibonacci hashing constant


def djb2_hash(s: str) -> int:
    """DJB2 hash function for strings. Returns uint32."""
    h = 5381
    for c in s:
        h = ((h << 5) + h + ord(c)) & 0xFFFFFFFF
    return h


def layer_seed(layer_idx: int, model_name: str = "") -> int:
    """Generate a deterministic per-layer seed.

    Uses golden ratio hashing for good distribution across layers,
    combined with model name hash to prevent cross-model collisions.

    Aligned with kernel implementation (ggml/Metal/CUDA).

    seed = layer_idx * GOLDEN_RATIO + djb2(model_name)
    """
    model_hash = djb2_hash(model_name) if model_name else 0
    return ((layer_idx * GOLDEN_RATIO) + model_hash) & 0xFFFFFFFF


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant compression."""
    head_dim: int = 128       # d: head dimension
    total_bits: int = 3       # b: total bits per coordinate (2-7)
    seed: int = 42            # Random seed for rotation and projection
    use_fast_rotation: bool = True  # Use WHT-based rotation (O(d·log d))
    use_qjl: bool = True      # Enable QJL correction (1 bit). False = pure MSE.

    def __post_init__(self):
        assert self.total_bits >= 2, f"total_bits must be >= 2, got {self.total_bits}"

    @property
    def mse_bits(self) -> int:
        """Bits allocated to MSE stage."""
        if self.use_qjl:
            return self.total_bits - 1
        return self.total_bits

    @property
    def qjl_bits(self) -> int:
        """Bits allocated to QJL stage."""
        return 1 if self.use_qjl else 0


class TurboQuantCompressor:
    """TurboQuant compressor/decompressor for KV cache vectors."""

    def __init__(self, config: TurboQuantConfig):
        self.config = config
        self.d = config.head_dim
        self.b = config.total_bits

        # Rotation matrix (or signs for fast rotation)
        if config.use_fast_rotation:
            assert (self.d & (self.d - 1)) == 0, "Fast rotation requires d to be power of 2"
            self.rotation_signs = generate_random_signs(self.d, seed=config.seed)
            self.Pi = None
        else:
            self.Pi = generate_random_orthogonal(self.d, seed=config.seed)
            self.rotation_signs = None

        # Lloyd-Max codebook for MSE stage
        self.codebook = lloyd_max(self.d, config.mse_bits)

        # QJL projection matrix S ∈ ℝ^{d×d} (only when QJL enabled)
        if config.use_qjl:
            qjl_rng = np.random.default_rng(config.seed + 1000)
            self.S = qjl_rng.standard_normal((self.d, self.d))
        else:
            self.S = None

    def _rotate(self, x: np.ndarray) -> np.ndarray:
        """Apply rotation to vector."""
        if self.config.use_fast_rotation:
            return fast_random_rotation(x, self.rotation_signs)
        else:
            return self.Pi @ x

    def _rotate_inverse(self, y: np.ndarray) -> np.ndarray:
        """Apply inverse rotation."""
        if self.config.use_fast_rotation:
            return fast_random_rotation_inverse(y, self.rotation_signs)
        else:
            return self.Pi.T @ y

    def compress(self, x: np.ndarray) -> CompressedKV:
        """Compress a single KV vector.

        PolarQuant approach: normalize to unit sphere, rotate, quantize with
        codebook trained on the unit-sphere coordinate distribution N(0, 1/d).
        Store the norm separately for reconstruction.

        Args:
            x: Input vector, shape (d,)

        Returns:
            CompressedKV object
        """
        assert x.shape == (self.d,), f"Expected shape ({self.d},), got {x.shape}"

        # Stage 1: MSE quantization (PolarQuant)
        x_norm = np.linalg.norm(x)
        if x_norm < 1e-12:
            indices = np.zeros(self.d, dtype=np.uint8)
            mse_packed = pack_mse_indices(indices, self.config.mse_bits)
            qjl_packed = pack_qjl_bits(np.ones(self.d)) if self.config.use_qjl else np.array([], dtype=np.uint8)
            return CompressedKV(
                mse_packed=mse_packed,
                qjl_packed=qjl_packed,
                gamma=0.0,
                mse_bits=self.config.mse_bits,
                d=self.d,
                x_norm=0.0,
            )

        x_hat = x / x_norm                             # Normalize to unit sphere
        y = self._rotate(x_hat)                        # Rotate (coords ~ N(0, 1/d))
        indices = quantize_scalar(y, self.codebook)    # Quantize each coordinate

        mse_packed = pack_mse_indices(indices, self.config.mse_bits)

        if self.config.use_qjl:
            # Stage 2: QJL on residual
            y_hat = dequantize_scalar(indices, self.codebook)
            x_mse = x_norm * self._rotate_inverse(y_hat)
            r = x - x_mse
            gamma = np.linalg.norm(r)
            qjl_signs = np.sign(self.S @ r) if gamma > 1e-12 else np.ones(self.d)
            qjl_packed = pack_qjl_bits(qjl_signs)
        else:
            # Pure MSE mode — no QJL
            gamma = 0.0
            qjl_packed = np.array([], dtype=np.uint8)

        return CompressedKV(
            mse_packed=mse_packed,
            qjl_packed=qjl_packed,
            gamma=gamma,
            mse_bits=self.config.mse_bits,
            d=self.d,
            x_norm=x_norm,
        )

    def decompress(self, compressed: CompressedKV) -> np.ndarray:
        """Decompress a KV vector (full reconstruction).

        Args:
            compressed: CompressedKV object

        Returns:
            Reconstructed vector, shape (d,)
        """
        indices = compressed.get_mse_indices()
        x_norm = float(compressed.x_norm)

        # Stage 1: MSE reconstruction (with original norm)
        y_hat = dequantize_scalar(indices, self.codebook)
        x_mse = x_norm * self._rotate_inverse(y_hat)

        if self.config.use_qjl:
            # Stage 2: QJL residual reconstruction
            qjl_signs = compressed.get_qjl_signs()
            gamma = float(compressed.gamma)
            scale = math.sqrt(math.pi / 2) / self.d
            x_qjl = scale * gamma * (self.S.T @ qjl_signs)
            return x_mse + x_qjl

        return x_mse

    def inner_product(self, q: np.ndarray, compressed: CompressedKV) -> float:
        """Compute ⟨q, x̃⟩ without full decompression.

        More efficient than decompress() + dot() when only the
        inner product is needed (e.g., attention logits).

        Args:
            q: Query vector, shape (d,)
            compressed: Compressed KV entry

        Returns:
            Approximate inner product ⟨q, x̃⟩
        """
        indices = compressed.get_mse_indices()
        x_norm = float(compressed.x_norm)

        # MSE component: x_norm · ⟨q, Πᵀỹ⟩ = x_norm · ⟨Πq, ỹ⟩
        q_rot = self._rotate(q)
        y_hat = dequantize_scalar(indices, self.codebook)
        ip_mse = x_norm * np.dot(q_rot, y_hat)

        if self.config.use_qjl:
            # QJL component: ⟨q, (√(π/2)/d)·γ·Sᵀ·z⟩ = (√(π/2)/d)·γ·⟨Sq, z⟩
            qjl_signs = compressed.get_qjl_signs()
            gamma = float(compressed.gamma)
            q_proj = self.S @ q
            ip_qjl = (math.sqrt(math.pi / 2) / self.d) * gamma * np.dot(q_proj, qjl_signs)
            return ip_mse + ip_qjl

        return ip_mse

    def compress_batch(self, X: np.ndarray) -> list[CompressedKV]:
        """Compress a batch of KV vectors.

        Args:
            X: Input matrix, shape (seq_len, d)

        Returns:
            List of CompressedKV objects
        """
        return [self.compress(X[i]) for i in range(X.shape[0])]

    def decompress_batch(self, compressed_list: list[CompressedKV]) -> np.ndarray:
        """Decompress a batch of KV vectors.

        Returns:
            Matrix of shape (seq_len, d)
        """
        return np.stack([self.decompress(c) for c in compressed_list])
