"""
HuggingFace Transformers integration for TurboQuant KV cache compression.

Provides a drop-in replacement for the default DynamicCache that compresses
KV entries using TurboQuant. Usage:

    from turboquant.hf_integration import TurboQuantCache

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
    cache = TurboQuantCache(total_bits=3, head_dim=128)
    output = model.generate(input_ids, past_key_values=cache)
"""

import torch
import numpy as np
import math
from typing import Optional, Tuple, List

from .turboquant import TurboQuantConfig, TurboQuantCompressor
from .bitpack import CompressedKV

try:
    from transformers.cache_utils import Cache
    HAS_CACHE_BASE = True
except ImportError:
    HAS_CACHE_BASE = False


class TurboQuantCacheLayer:
    """Compressed KV cache for a single layer.

    Stores keys and values in TurboQuant compressed format.
    Decompresses on-the-fly when accessed for attention computation.

    Implements the CacheLayerMixin-compatible interface expected by
    transformers >= 5.x (get_mask_sizes, get_seq_length, get_max_cache_shape).
    """

    is_sliding = False

    def __init__(self, compressor: TurboQuantCompressor, num_heads: int,
                 per_channel_scaling: bool = False):
        self.compressor = compressor
        self.num_heads = num_heads
        self.per_channel_scaling = per_channel_scaling
        self._keys: List[List[CompressedKV]] = [[] for _ in range(num_heads)]
        self._values: List[List[CompressedKV]] = [[] for _ in range(num_heads)]
        # Per-channel scale factors: shape (num_heads, head_dim), fp16
        # Computed from the first batch (prefill) of KV vectors
        self._k_scales: Optional[np.ndarray] = None
        self._v_scales: Optional[np.ndarray] = None
        self._seen_tokens = 0
        self.is_initialized = False

    @property
    def seq_length(self) -> int:
        if self._keys[0]:
            return len(self._keys[0])
        return 0

    def get_seq_length(self) -> int:
        return self.seq_length

    def get_mask_sizes(self, query_length: int) -> Tuple[int, int]:
        kv_length = self.get_seq_length() + query_length
        kv_offset = 0
        return kv_length, kv_offset

    def get_max_cache_shape(self) -> int:
        return -1  # No fixed max

    @property
    def max_batch_size(self) -> int:
        return 1

    @property
    def max_cache_len(self) -> int:
        return self.seq_length

    def _compute_channel_scales(self, data_np: np.ndarray) -> np.ndarray:
        """Compute per-channel scale factors from a batch of vectors.

        Args:
            data_np: shape (num_heads, seq_len, head_dim)

        Returns:
            scales: shape (num_heads, head_dim), per-channel max abs values.
                    Minimum 1e-6 to avoid division by zero.
        """
        # max abs per dimension across the sequence
        scales = np.max(np.abs(data_np), axis=1)  # (num_heads, head_dim)
        scales = np.maximum(scales, 1e-6)
        return scales

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress and store new KV entries, return decompressed full cache.

        Args:
            key_states: shape (batch, num_heads, seq_len, head_dim)
            value_states: shape (batch, num_heads, seq_len, head_dim)

        Returns:
            (all_keys, all_values) decompressed tensors
        """
        assert key_states.shape[0] == 1, "Only batch_size=1 supported"
        self.is_initialized = True

        batch, num_heads, seq_len, head_dim = key_states.shape
        k_np = key_states[0].detach().cpu().float().numpy()
        v_np = value_states[0].detach().cpu().float().numpy()

        # Compute per-channel scales on the first batch (prefill)
        if self.per_channel_scaling and self._k_scales is None:
            self._k_scales = self._compute_channel_scales(k_np)
            self._v_scales = self._compute_channel_scales(v_np)

        for h in range(num_heads):
            for s in range(seq_len):
                k_vec = k_np[h, s]
                v_vec = v_np[h, s]
                if self.per_channel_scaling and self._k_scales is not None:
                    k_vec = k_vec / self._k_scales[h]
                    v_vec = v_vec / self._v_scales[h]
                self._keys[h].append(self.compressor.compress(k_vec))
                self._values[h].append(self.compressor.compress(v_vec))

        self._seen_tokens += seq_len

        device = key_states.device
        dtype = key_states.dtype
        all_keys = self._get_keys(device, dtype)
        all_values = self._get_values(device, dtype)
        return all_keys, all_values

    def _get_keys(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.seq_length == 0:
            return torch.zeros(1, self.num_heads, 0, self.compressor.d,
                             device=device, dtype=dtype)
        keys_np = np.zeros((self.num_heads, self.seq_length, self.compressor.d))
        for h in range(self.num_heads):
            for s in range(self.seq_length):
                vec = self.compressor.decompress(self._keys[h][s])
                if self.per_channel_scaling and self._k_scales is not None:
                    vec = vec * self._k_scales[h]
                keys_np[h, s] = vec
        return torch.from_numpy(keys_np).unsqueeze(0).to(device=device, dtype=dtype)

    def _get_values(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.seq_length == 0:
            return torch.zeros(1, self.num_heads, 0, self.compressor.d,
                             device=device, dtype=dtype)
        values_np = np.zeros((self.num_heads, self.seq_length, self.compressor.d))
        for h in range(self.num_heads):
            for s in range(self.seq_length):
                vec = self.compressor.decompress(self._values[h][s])
                if self.per_channel_scaling and self._v_scales is not None:
                    vec = vec * self._v_scales[h]
                values_np[h, s] = vec
        return torch.from_numpy(values_np).unsqueeze(0).to(device=device, dtype=dtype)


_CacheBase = Cache if HAS_CACHE_BASE else object


class TurboQuantCache(_CacheBase):
    """Drop-in replacement for HuggingFace DynamicCache with TurboQuant compression.

    Inherits from transformers.Cache when available for full compatibility
    with transformers >= 5.x masking and generation infrastructure.
    """

    def __init__(self, total_bits: int = 3, head_dim: int = 128, seed: int = 42,
                 per_channel_scaling: bool = False):
        if HAS_CACHE_BASE:
            super().__init__(layer_class_to_replicate=TurboQuantCacheLayer)
        self.config = TurboQuantConfig(
            head_dim=head_dim,
            total_bits=total_bits,
            seed=seed,
        )
        self.compressor = TurboQuantCompressor(self.config)
        self.per_channel_scaling = per_channel_scaling
        self.layers: List[TurboQuantCacheLayer] = []
        self._seen_tokens = 0

    def _ensure_layer(self, layer_idx: int, num_heads: int):
        """Lazily create cache layers as needed."""
        while len(self.layers) <= layer_idx:
            self.layers.append(TurboQuantCacheLayer(
                self.compressor, num_heads,
                per_channel_scaling=self.per_channel_scaling))

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Store new KV and return full (decompressed) KV for attention.

        Matches the DynamicCache.update() / Cache.update() signature.
        """
        num_heads = key_states.shape[1]
        self._ensure_layer(layer_idx, num_heads)

        all_keys, all_values = self.layers[layer_idx].update(key_states, value_states)

        if layer_idx == 0:
            self._seen_tokens += key_states.shape[2]

        return all_keys, all_values

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx < len(self.layers):
            return self.layers[layer_idx].seq_length
        return 0

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        if layer_idx < len(self.layers):
            return self.layers[layer_idx].get_max_cache_shape()
        return -1

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> Tuple[int, int]:
        if layer_idx >= len(self.layers):
            return query_length, 0
        return self.layers[layer_idx].get_mask_sizes(query_length)

    @property
    def is_initialized(self) -> bool:
        return len(self.layers) > 0 and all(l.is_initialized for l in self.layers)

    @property
    def is_sliding(self) -> list:
        return [False] * len(self.layers)

    @property
    def max_batch_size(self) -> int:
        return 1

    @property
    def max_cache_len(self) -> int:
        if not self.layers:
            return 0
        return max(l.seq_length for l in self.layers)

    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx >= len(self.layers):
            raise IndexError(f"Layer {layer_idx} not in cache (have {len(self.layers)} layers)")
        layer = self.layers[layer_idx]
        keys = layer._get_keys(torch.device('cpu'), torch.float32)
        values = layer._get_values(torch.device('cpu'), torch.float32)
        return keys, values

    def __iter__(self):
        for layer_idx in range(len(self.layers)):
            yield self[layer_idx]

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple(
            (layer._get_keys(torch.device('cpu'), torch.float32),
             layer._get_values(torch.device('cpu'), torch.float32))
            for layer in self.layers
        )

    def memory_usage_bytes(self) -> int:
        total = 0
        for layer in self.layers:
            for h in range(layer.num_heads):
                for ckv in layer._keys[h]:
                    total += ckv.total_bytes
                for ckv in layer._values[h]:
                    total += ckv.total_bytes
        return total

    def memory_savings_ratio(self, head_dim: int = 128) -> float:
        if not self.layers or self.layers[0].seq_length == 0:
            return 1.0
        fp16_bytes = sum(
            2 * layer.num_heads * layer.seq_length * head_dim * 2
            for layer in self.layers
        )
        return fp16_bytes / max(self.memory_usage_bytes(), 1)
