"""
Bit packing utilities for TurboQuant compressed KV cache.

Formats:
- turbo2: 1 bit MSE + 1 bit QJL = 2 bits/coordinate
- turbo3: 2 bits MSE + 1 bit QJL = 3 bits/coordinate
- turbo4: 3 bits MSE + 1 bit QJL = 4 bits/coordinate

Plus one fp16 gamma (residual norm) per vector.
"""

import numpy as np
from typing import Tuple


def pack_mse_indices(indices: np.ndarray, bits: int) -> np.ndarray:
    """Pack MSE quantization indices into bytes.

    Args:
        indices: Array of shape (d,) with values in [0, 2^bits - 1]
        bits: Bits per index (1, 2, or 3)

    Returns:
        Packed byte array
    """
    d = len(indices)
    if bits == 1:
        # Pack 8 indices per byte
        n_bytes = (d + 7) // 8
        packed = np.zeros(n_bytes, dtype=np.uint8)
        for i in range(d):
            packed[i // 8] |= (indices[i] & 0x1) << (i % 8)
        return packed

    elif bits == 2:
        # Pack 4 indices per byte
        n_bytes = (d + 3) // 4
        packed = np.zeros(n_bytes, dtype=np.uint8)
        for i in range(d):
            packed[i // 4] |= (indices[i] & 0x3) << (2 * (i % 4))
        return packed

    elif bits == 3:
        # Pack into groups: 8 indices in 3 bytes (24 bits)
        n_groups = (d + 7) // 8
        packed = np.zeros(n_groups * 3, dtype=np.uint8)
        for i in range(d):
            bit_offset = i * 3
            byte_idx = bit_offset // 8
            bit_idx = bit_offset % 8
            val = indices[i] & 0x7
            packed[byte_idx] |= (val << bit_idx) & 0xFF
            if bit_idx > 5:  # Overflows into next byte
                packed[byte_idx + 1] |= val >> (8 - bit_idx)
        return packed

    elif bits == 4:
        # Pack 2 indices per byte
        n_bytes = (d + 1) // 2
        packed = np.zeros(n_bytes, dtype=np.uint8)
        for i in range(d):
            packed[i // 2] |= (indices[i] & 0xF) << (4 * (i % 2))
        return packed

    elif bits == 5:
        # Pack using generic bit-stream approach: 5 bits per index
        total_bits_needed = d * 5
        n_bytes = (total_bits_needed + 7) // 8
        packed = np.zeros(n_bytes, dtype=np.uint8)
        for i in range(d):
            bit_offset = i * 5
            byte_idx = bit_offset // 8
            bit_idx = bit_offset % 8
            val = indices[i] & 0x1F
            packed[byte_idx] |= (val << bit_idx) & 0xFF
            if bit_idx > 3:  # Overflows into next byte
                if byte_idx + 1 < n_bytes:
                    packed[byte_idx + 1] |= val >> (8 - bit_idx)
        return packed

    elif bits == 6:
        # Pack 4 indices per 3 bytes (24 bits), matching ggml TQKV_6 layout
        assert d % 4 == 0, f"6-bit packing requires d divisible by 4, got {d}"
        n_groups = d // 4
        packed = np.zeros(n_groups * 3, dtype=np.uint8)
        for g in range(n_groups):
            i0 = indices[g * 4 + 0] & 0x3F
            i1 = indices[g * 4 + 1] & 0x3F
            i2 = indices[g * 4 + 2] & 0x3F
            i3 = indices[g * 4 + 3] & 0x3F
            packed[g * 3 + 0] = i0 | ((i1 & 0x03) << 6)
            packed[g * 3 + 1] = ((i1 >> 2) & 0x0F) | ((i2 & 0x0F) << 4)
            packed[g * 3 + 2] = ((i2 >> 4) & 0x03) | (i3 << 2)
        return packed

    else:
        raise ValueError(f"Unsupported bits: {bits}")


def unpack_mse_indices(packed: np.ndarray, d: int, bits: int) -> np.ndarray:
    """Unpack MSE quantization indices from bytes.

    Args:
        packed: Packed byte array
        d: Number of coordinates
        bits: Bits per index

    Returns:
        indices: Array of shape (d,)
    """
    indices = np.zeros(d, dtype=np.uint8)

    if bits == 1:
        for i in range(d):
            indices[i] = (packed[i // 8] >> (i % 8)) & 0x1

    elif bits == 2:
        for i in range(d):
            indices[i] = (packed[i // 4] >> (2 * (i % 4))) & 0x3

    elif bits == 3:
        for i in range(d):
            bit_offset = i * 3
            byte_idx = bit_offset // 8
            bit_idx = bit_offset % 8
            val = (packed[byte_idx] >> bit_idx)
            if bit_idx > 5 and byte_idx + 1 < len(packed):
                val |= packed[byte_idx + 1] << (8 - bit_idx)
            indices[i] = val & 0x7

    elif bits == 4:
        for i in range(d):
            indices[i] = (packed[i // 2] >> (4 * (i % 2))) & 0xF

    elif bits == 5:
        for i in range(d):
            bit_offset = i * 5
            byte_idx = bit_offset // 8
            bit_idx = bit_offset % 8
            val = (packed[byte_idx] >> bit_idx)
            if bit_idx > 3 and byte_idx + 1 < len(packed):
                val |= packed[byte_idx + 1] << (8 - bit_idx)
            indices[i] = val & 0x1F

    elif bits == 6:
        assert d % 4 == 0, f"6-bit unpacking requires d divisible by 4, got {d}"
        n_groups = d // 4
        for g in range(n_groups):
            b0 = packed[g * 3 + 0]
            b1 = packed[g * 3 + 1]
            b2 = packed[g * 3 + 2]
            indices[g * 4 + 0] = b0 & 0x3F
            indices[g * 4 + 1] = ((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2)
            indices[g * 4 + 2] = ((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4)
            indices[g * 4 + 3] = (b2 >> 2) & 0x3F

    else:
        raise ValueError(f"Unsupported bits: {bits}")

    return indices


def pack_qjl_bits(signs: np.ndarray) -> np.ndarray:
    """Pack QJL sign bits (+1/-1) into bytes.

    Args:
        signs: Array of shape (d,) with values in {-1, +1}

    Returns:
        Packed byte array (d/8 bytes)
    """
    d = len(signs)
    n_bytes = (d + 7) // 8
    packed = np.zeros(n_bytes, dtype=np.uint8)
    # Encode: +1 → 1, -1 → 0
    bits = (signs > 0).astype(np.uint8)
    for i in range(d):
        packed[i // 8] |= bits[i] << (i % 8)
    return packed


def unpack_qjl_bits(packed: np.ndarray, d: int) -> np.ndarray:
    """Unpack QJL sign bits from bytes.

    Returns:
        signs: Array of shape (d,) with values in {-1, +1}
    """
    signs = np.zeros(d, dtype=np.float64)
    for i in range(d):
        bit = (packed[i // 8] >> (i % 8)) & 0x1
        signs[i] = 1.0 if bit else -1.0
    return signs


class CompressedKV:
    """Compressed KV cache entry for a single vector."""

    def __init__(self, mse_packed: np.ndarray, qjl_packed: np.ndarray,
                 gamma: float, mse_bits: int, d: int, x_norm: float = 1.0):
        self.mse_packed = mse_packed
        self.qjl_packed = qjl_packed
        self.gamma = np.float32(gamma)
        self.x_norm = np.float32(x_norm)
        self.mse_bits = mse_bits
        self.d = d

    @property
    def total_bits(self) -> int:
        """Total bits per coordinate."""
        return self.mse_bits + 1  # MSE + QJL

    @property
    def total_bytes(self) -> int:
        """Total storage for this compressed vector."""
        return len(self.mse_packed) + len(self.qjl_packed) + 8  # +4 fp32 gamma + 4 fp32 x_norm

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs fp16 storage."""
        original_bytes = self.d * 2  # fp16
        return original_bytes / self.total_bytes

    def get_mse_indices(self) -> np.ndarray:
        return unpack_mse_indices(self.mse_packed, self.d, self.mse_bits)

    def get_qjl_signs(self) -> np.ndarray:
        return unpack_qjl_bits(self.qjl_packed, self.d)
