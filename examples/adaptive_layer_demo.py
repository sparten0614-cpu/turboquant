#!/usr/bin/env python3
"""
Adaptive Layer Selection Demo
==============================

Demonstrates TurboQuant's auto-detection of outlier layers and the
impact of skipping them on compression quality.

Requirements:
    pip install turboquant[hf]

Usage:
    python examples/adaptive_layer_demo.py
"""

import sys
import os
import time
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from turboquant.calibration import detect_outlier_layers, profile_kv_cache
from turboquant.turboquant import TurboQuantConfig, TurboQuantCompressor
from turboquant.hf_integration import TurboQuantCacheLayer


# ─── Config ───────────────────────────────────────────────────────────
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cpu"  # Works on any machine; change to "mps" or "cuda" if available
CONTEXT_TOKENS = 64
EVAL_TOKENS = 32
TOTAL_BITS = 4


def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        device_map=DEVICE,
    )
    model.eval()
    head_dim = getattr(model.config, "head_dim",
                       model.config.hidden_size // model.config.num_attention_heads)
    print(f"  {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params, "
          f"head_dim={head_dim}, "
          f"kv_heads={model.config.num_key_value_heads}, "
          f"layers={model.config.num_hidden_layers}")
    return model, tokenizer, head_dim


def load_eval_data(tokenizer, max_tokens):
    print("Loading WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=max_tokens)["input_ids"].to(DEVICE)
    print(f"  {input_ids.shape[1]} tokens")
    return input_ids


# ─── Part 1: Outlier Layer Detection ──────────────────────────────────

def demo_detection(model, tokenizer):
    print("\n" + "=" * 60)
    print("Part 1: Outlier Layer Detection")
    print("=" * 60)

    print("\nRunning detect_outlier_layers(threshold=3.0)...")
    outliers = detect_outlier_layers(model, tokenizer, n_tokens=32, threshold=3.0)
    print(f"  Outlier layers: {outliers if outliers else '[] (none)'}")

    if not outliers:
        print("  TinyLlama has uniform K distributions — no outliers detected.")
        print("  (Qwen2.5-3B would return [0, 27] here)")

    return outliers


# ─── Part 2: Per-Layer KV Cache Profile ───────────────────────────────

def demo_profile(model, tokenizer):
    print("\n" + "=" * 60)
    print("Part 2: Per-Layer KV Cache Profile")
    print("=" * 60)

    profile = profile_kv_cache(model, tokenizer, n_tokens=32)

    print(f"\nModel: {MODEL_NAME} ({profile['num_layers']} layers)")
    print(f"Median K_max: {profile['median_k_max']:.2f}")
    print(f"Outliers at 3x: {profile['outliers_3x'] or 'none'}")
    print(f"Outliers at 5x: {profile['outliers_5x'] or 'none'}")

    print(f"\n{'Layer':>5}  {'K_max':>8}  {'K_norm':>8}  {'V_max':>8}  {'Flag':>6}")
    print("-" * 45)
    for i in range(profile["num_layers"]):
        flag = " ***" if i in profile["outliers_3x"] else ""
        print(f"{i:5d}  {profile['k_max'][i]:8.2f}  "
              f"{profile['k_mean_norm'][i]:8.2f}  "
              f"{profile['v_max'][i]:8.2f}  {flag}")

    return profile


# ─── Part 3: Simulated Outlier Injection ──────────────────────────────

def demo_simulated_outlier(model, tokenizer):
    print("\n" + "=" * 60)
    print("Part 3: Simulated Outlier Layer (what Qwen looks like)")
    print("=" * 60)

    print("\nInjecting outlier into TinyLlama's KV cache to simulate Qwen...")

    # Run forward pass
    text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(inputs["input_ids"], use_cache=True)
        cache = outputs.past_key_values

    # Collect original K_max per layer
    original_k_max = []
    for layer in cache.layers:
        original_k_max.append(layer.keys.abs().max().item())

    # Inject outlier: multiply layer 0's keys by 10x (simulating Qwen-like outlier)
    cache.layers[0].keys = cache.layers[0].keys * 10.0

    injected_k_max = []
    for layer in cache.layers:
        injected_k_max.append(layer.keys.abs().max().item())

    # Now run detection on the modified cache
    # (We'll do it manually since detect_outlier_layers runs its own forward pass)
    k_maxes = np.array(injected_k_max)
    median = np.median(k_maxes)
    detected = [i for i, km in enumerate(k_maxes) if km > 3.0 * median]

    print(f"  Original Layer 0 K_max:  {original_k_max[0]:.2f}")
    print(f"  Injected Layer 0 K_max:  {injected_k_max[0]:.2f} (10x)")
    print(f"  Median K_max:            {median:.2f}")
    print(f"  3x threshold:            {3.0 * median:.2f}")
    print(f"  Detected outliers:       {detected}")

    if 0 in detected:
        print("  Layer 0 correctly detected as outlier after injection.")
    else:
        print("  WARNING: Layer 0 not detected (threshold may need adjustment)")


# ─── Part 4: Compression With/Without Outlier Skip ────────────────────

def compress_cache(cache, head_dim, total_bits, skip_layers=None):
    """Compress a DynamicCache, optionally skipping certain layers."""
    if skip_layers is None:
        skip_layers = set()
    else:
        skip_layers = set(skip_layers)

    config = TurboQuantConfig(head_dim=head_dim, total_bits=total_bits)
    compressor = TurboQuantCompressor(config)

    total_mse = 0.0
    total_count = 0

    for layer_idx, layer in enumerate(cache.layers):
        if not layer.is_initialized:
            continue
        if layer_idx in skip_layers:
            continue  # Keep at FP16

        keys = layer.keys[0].float().numpy()  # (heads, seq, head_dim)
        for h in range(keys.shape[0]):
            for s in range(keys.shape[1]):
                x = keys[h, s].astype(np.float64)
                compressed = compressor.compress(x)
                x_hat = compressor.decompress(compressed)
                total_mse += np.sum((x - x_hat) ** 2)
                total_count += 1

    avg_mse = total_mse / max(total_count, 1)
    return avg_mse, total_count


def eval_perplexity(model, input_ids, context_len):
    """Baseline perplexity (no compression)."""
    with torch.no_grad():
        outputs = model(input_ids[:, :context_len], use_cache=True)
        cache = outputs.past_key_values

        total_loss = 0.0
        for i in range(context_len, input_ids.shape[1]):
            prev = input_ids[:, i - 1:i]
            target = input_ids[:, i:i + 1]
            out = model(prev, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            loss = torch.nn.functional.cross_entropy(out.logits[:, -1, :], target.squeeze(0))
            total_loss += loss.item()

    avg_loss = total_loss / (input_ids.shape[1] - context_len)
    return math.exp(avg_loss)


def demo_compression(model, tokenizer, head_dim):
    print("\n" + "=" * 60)
    print("Part 4: Compression Quality — With vs Without Outlier Skip")
    print("=" * 60)

    text = "The quick brown fox jumps over the lazy dog. " * 8
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=64).to(DEVICE)

    with torch.no_grad():
        outputs = model(inputs["input_ids"], use_cache=True)
        cache = outputs.past_key_values

    # Inject outlier to layer 0 (simulate Qwen)
    cache.layers[0].keys = cache.layers[0].keys * 10.0

    num_layers = len(cache.layers)
    print(f"\n  Simulated outlier: Layer 0 keys scaled 10x")
    print(f"  Compressing {num_layers} layers at {TOTAL_BITS}-bit...")

    # Compress all layers
    mse_all, count_all = compress_cache(cache, head_dim, TOTAL_BITS, skip_layers=[])
    # Compress with layer 0 skipped
    mse_skip, count_skip = compress_cache(cache, head_dim, TOTAL_BITS, skip_layers=[0])

    skipped_pct = (1 / num_layers) * 100

    print(f"\n  {'Config':<30} {'Avg MSE':>12} {'Vectors':>10}")
    print("  " + "-" * 55)
    print(f"  {'Uniform (all layers)':<30} {mse_all:>12.4f} {count_all:>10d}")
    print(f"  {'Skip Layer 0 (FP16)':<30} {mse_skip:>12.4f} {count_skip:>10d}")
    print(f"\n  MSE reduction: {(1 - mse_skip / mse_all) * 100:.1f}%")
    print(f"  Extra FP16 memory: ~{skipped_pct:.1f}% ({1}/{num_layers} layers)")


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    print("TurboQuant Adaptive Layer Selection Demo")
    print("=" * 60)

    model, tokenizer, head_dim = load_model()

    # Part 1: Detection
    outliers = demo_detection(model, tokenizer)

    # Part 2: Profile
    profile = demo_profile(model, tokenizer)

    # Part 3: Simulated outlier
    demo_simulated_outlier(model, tokenizer)

    # Part 4: Compression comparison
    demo_compression(model, tokenizer, head_dim)

    print("\n" + "=" * 60)
    print("Demo complete.")
    print()
    print("Key takeaway: Adaptive Layer Selection detects outlier layers")
    print("automatically and keeps them at FP16. For Qwen2.5-3B, skipping")
    print("just 2/36 layers reduces PPL degradation from +4.4% to +0.04%")
    print("at only 5.6% extra memory.")
    print("=" * 60)


if __name__ == "__main__":
    main()
