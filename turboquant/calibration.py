"""
Adaptive Layer Selection: auto-detect outlier layers for FP16 fallback.

Outlier layers have extreme Key magnitudes that cause quantization error
to dominate. This module profiles a model's KV cache and identifies which
layers should be kept at FP16 while the rest use TurboQuant compression.

Requires: torch, transformers (install with pip install turboquant[hf])
"""

import torch
import numpy as np
from typing import List, Optional


def detect_outlier_layers(
    model,
    tokenizer,
    n_tokens: int = 32,
    threshold: float = 3.0,
    device: Optional[str] = None,
) -> List[int]:
    """Detect outlier layers by profiling Key cache magnitudes.

    Runs a short forward pass and identifies layers where the maximum
    Key absolute value exceeds threshold * median across all layers.

    Args:
        model: HuggingFace CausalLM model (already loaded).
        tokenizer: Corresponding tokenizer.
        n_tokens: Number of tokens for calibration (default 32, more is
                  slightly more accurate but slower).
        threshold: Multiplier for median K_max. Layers with K_max >
                   threshold * median are flagged as outliers.
                   Default 3.0 works well empirically.
        device: Device override. If None, uses model's device.

    Returns:
        List of outlier layer indices (sorted). Empty list if no outliers.

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        >>> outliers = detect_outlier_layers(model, tokenizer)
        >>> print(outliers)  # e.g., [0, 27]
    """
    if device is None:
        device = next(model.parameters()).device

    # Generate calibration input
    text = "The quick brown fox jumps over the lazy dog. " * 4
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=n_tokens
    )
    input_ids = inputs["input_ids"].to(device)

    # Forward pass to populate KV cache
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        cache = outputs.past_key_values

    # Collect per-layer K_max
    k_maxes = []
    for layer in cache.layers if hasattr(cache, 'layers') else cache:
        if hasattr(layer, 'keys'):
            # DynamicCache style (transformers >= 5.x)
            keys = layer.keys
        elif isinstance(layer, tuple) and len(layer) == 2:
            # Legacy tuple style
            keys = layer[0]
        else:
            continue

        k_max = keys.abs().max().item()
        k_maxes.append(k_max)

    if not k_maxes:
        return []

    k_maxes = np.array(k_maxes)
    median_k_max = np.median(k_maxes)

    if median_k_max < 1e-6:
        return []

    # Flag layers exceeding threshold * median
    outlier_threshold = threshold * median_k_max
    outliers = [i for i, km in enumerate(k_maxes) if km > outlier_threshold]

    return sorted(outliers)


def profile_kv_cache(
    model,
    tokenizer,
    n_tokens: int = 32,
    device: Optional[str] = None,
) -> dict:
    """Profile KV cache statistics per layer.

    Returns detailed statistics for analysis and threshold tuning.

    Args:
        model: HuggingFace CausalLM model.
        tokenizer: Corresponding tokenizer.
        n_tokens: Calibration tokens.
        device: Device override.

    Returns:
        Dict with keys:
            - k_max: list of per-layer Key max abs values
            - k_mean_norm: list of per-layer mean Key vector norms
            - v_max: list of per-layer Value max abs values
            - num_layers: total layer count
            - outliers_3x: layers with K_max > 3x median
            - outliers_5x: layers with K_max > 5x median
    """
    if device is None:
        device = next(model.parameters()).device

    text = "The quick brown fox jumps over the lazy dog. " * 4
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=n_tokens
    )
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        cache = outputs.past_key_values

    k_maxes = []
    k_mean_norms = []
    v_maxes = []

    for layer in cache.layers if hasattr(cache, 'layers') else cache:
        if hasattr(layer, 'keys'):
            keys, values = layer.keys, layer.values
        elif isinstance(layer, tuple) and len(layer) == 2:
            keys, values = layer[0], layer[1]
        else:
            continue

        k_maxes.append(keys.abs().max().item())
        k_mean_norms.append(keys.float().norm(dim=-1).mean().item())
        v_maxes.append(values.abs().max().item())

    k_maxes = np.array(k_maxes)
    median_k = np.median(k_maxes)

    return {
        "k_max": k_maxes.tolist(),
        "k_mean_norm": k_mean_norms,
        "v_max": v_maxes,
        "num_layers": len(k_maxes),
        "median_k_max": float(median_k),
        "outliers_3x": [i for i, km in enumerate(k_maxes) if km > 3.0 * median_k],
        "outliers_5x": [i for i, km in enumerate(k_maxes) if km > 5.0 * median_k],
    }
