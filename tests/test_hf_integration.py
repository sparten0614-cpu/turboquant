"""Test TurboQuantCache with HuggingFace transformers."""

import torch
import numpy as np

from turboquant.hf_integration import TurboQuantCache


def test_cache_basic():
    """Test basic cache operations with random tensors."""
    cache = TurboQuantCache(total_bits=3, head_dim=128, seed=42)

    # Simulate layer 0 update
    batch, num_heads, seq_len, head_dim = 1, 4, 8, 128
    key_states = torch.randn(batch, num_heads, seq_len, head_dim)
    value_states = torch.randn(batch, num_heads, seq_len, head_dim)

    all_keys, all_values = cache.update(key_states, value_states, layer_idx=0)

    assert all_keys.shape == (1, 4, 8, 128), f"Keys shape wrong: {all_keys.shape}"
    assert all_values.shape == (1, 4, 8, 128), f"Values shape wrong: {all_values.shape}"
    assert cache.get_seq_length(0) == 8
    assert cache.seen_tokens == 8
    print("PASS: cache_basic")


def test_cache_incremental():
    """Test incremental token appending (generation mode)."""
    cache = TurboQuantCache(total_bits=3, head_dim=128, seed=42)

    num_heads = 4
    head_dim = 128

    # Prefill: 16 tokens
    k = torch.randn(1, num_heads, 16, head_dim)
    v = torch.randn(1, num_heads, 16, head_dim)
    cache.update(k, v, layer_idx=0)
    assert cache.get_seq_length(0) == 16

    # Generate: 1 token at a time
    for step in range(4):
        k_new = torch.randn(1, num_heads, 1, head_dim)
        v_new = torch.randn(1, num_heads, 1, head_dim)
        all_k, all_v = cache.update(k_new, v_new, layer_idx=0)
        expected_len = 16 + step + 1
        assert all_k.shape[2] == expected_len, f"Step {step}: expected {expected_len}, got {all_k.shape[2]}"

    assert cache.get_seq_length(0) == 20
    assert cache.seen_tokens == 20
    print("PASS: cache_incremental")


def test_cache_multi_layer():
    """Test multi-layer cache."""
    cache = TurboQuantCache(total_bits=3, head_dim=128, seed=42)

    num_layers = 4
    num_heads = 4
    head_dim = 128

    for layer in range(num_layers):
        k = torch.randn(1, num_heads, 8, head_dim)
        v = torch.randn(1, num_heads, 8, head_dim)
        cache.update(k, v, layer_idx=layer)

    assert len(cache) == 4
    for layer in range(num_layers):
        assert cache.get_seq_length(layer) == 8
    print("PASS: cache_multi_layer")


def test_cache_quality():
    """Test that cached KV approximate original well."""
    cache = TurboQuantCache(total_bits=4, head_dim=128, seed=42)

    k_orig = torch.randn(1, 2, 4, 128)
    v_orig = torch.randn(1, 2, 4, 128)

    all_k, all_v = cache.update(k_orig, v_orig, layer_idx=0)

    # Compute cosine similarity per head per position
    for h in range(2):
        for s in range(4):
            k_cos = torch.nn.functional.cosine_similarity(
                k_orig[0, h, s].unsqueeze(0),
                all_k[0, h, s].unsqueeze(0)
            ).item()
            v_cos = torch.nn.functional.cosine_similarity(
                v_orig[0, h, s].unsqueeze(0),
                all_v[0, h, s].unsqueeze(0)
            ).item()
            # QJL adds noise to individual vectors (optimized for inner products)
            # Per-vector cos_sim threshold is relaxed
            assert k_cos > 0.5, f"Key cos_sim too low: {k_cos}"
            assert v_cos > 0.5, f"Value cos_sim too low: {v_cos}"

    print("PASS: cache_quality (turbo4 per-vector cos_sim > 0.5)")


def test_cache_memory_savings():
    """Test memory savings reporting."""
    cache = TurboQuantCache(total_bits=3, head_dim=128, seed=42)

    # 32 layers, 32 heads, 64 tokens
    for layer in range(32):
        k = torch.randn(1, 32, 64, 128)
        v = torch.randn(1, 32, 64, 128)
        cache.update(k, v, layer_idx=layer)

    compressed_bytes = cache.memory_usage_bytes()
    ratio = cache.memory_savings_ratio(head_dim=128)

    print(f"  Compressed: {compressed_bytes / 1e6:.1f} MB")
    print(f"  Savings ratio: {ratio:.1f}x")
    assert ratio > 4.0, f"Expected > 4x compression, got {ratio:.1f}x"
    print("PASS: cache_memory_savings")


def test_per_channel_scaling_functional():
    """Test per-channel scaling runs without errors.

    Note: PCS does NOT improve quality with PolarQuant's unit-sphere
    normalization. It conflicts with the existing normalization and can
    make things worse. The correct solution for high-norm models is to
    increase bit-width. This test verifies PCS doesn't crash.
    """
    batch, num_heads, seq_len, head_dim = 1, 2, 8, 128

    torch.manual_seed(42)
    k_orig = torch.randn(batch, num_heads, seq_len, head_dim)
    v_orig = torch.randn(batch, num_heads, seq_len, head_dim)

    cache = TurboQuantCache(total_bits=4, head_dim=128, seed=42,
                            per_channel_scaling=True)
    k_out, v_out = cache.update(k_orig, v_orig, layer_idx=0)

    assert k_out.shape == k_orig.shape
    assert v_out.shape == v_orig.shape
    assert not torch.any(torch.isnan(k_out))
    assert not torch.any(torch.isnan(v_out))
    print("PASS: per_channel_scaling_functional")


if __name__ == "__main__":
    print("=" * 60)
    print("TurboQuant HF Integration Tests")
    print("=" * 60)

    test_cache_basic()
    test_cache_incremental()
    test_cache_multi_layer()
    test_cache_quality()
    test_cache_memory_savings()
    test_per_channel_scaling_functional()

    print("=" * 60)
    print("ALL HF INTEGRATION TESTS PASSED")
    print("=" * 60)
