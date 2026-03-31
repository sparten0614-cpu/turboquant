"""Tests for Adaptive Layer Selection calibration."""

import pytest
import numpy as np

from turboquant.calibration import detect_outlier_layers, profile_kv_cache


@pytest.fixture(scope="module")
def tinyllama():
    """Load TinyLlama once for all tests in this module."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="cpu",
    )
    model.eval()
    return model, tokenizer


def test_detect_outlier_layers_tinyllama(tinyllama):
    """TinyLlama has no outlier layers — should return empty list."""
    model, tokenizer = tinyllama
    outliers = detect_outlier_layers(model, tokenizer, n_tokens=32, threshold=3.0)

    assert isinstance(outliers, list)
    assert all(isinstance(i, int) for i in outliers)
    # TinyLlama K norms are uniform (~5-30), no extreme outliers
    assert len(outliers) == 0, (
        f"TinyLlama should have no outlier layers at 3x threshold, got {outliers}"
    )


def test_detect_outlier_layers_returns_sorted(tinyllama):
    """Output should always be sorted."""
    model, tokenizer = tinyllama
    # Use very low threshold to force some "outliers" for testing
    outliers = detect_outlier_layers(model, tokenizer, n_tokens=32, threshold=1.0)
    assert outliers == sorted(outliers)


def test_profile_kv_cache(tinyllama):
    """Profile should return valid statistics for all layers."""
    model, tokenizer = tinyllama
    profile = profile_kv_cache(model, tokenizer, n_tokens=32)

    assert profile["num_layers"] == 22  # TinyLlama has 22 layers
    assert len(profile["k_max"]) == 22
    assert len(profile["k_mean_norm"]) == 22
    assert len(profile["v_max"]) == 22
    assert all(km > 0 for km in profile["k_max"])
    assert all(vm > 0 for vm in profile["v_max"])
    assert profile["median_k_max"] > 0


def test_profile_outlier_thresholds(tinyllama):
    """TinyLlama should have no outliers at either 3x or 5x threshold."""
    model, tokenizer = tinyllama
    profile = profile_kv_cache(model, tokenizer, n_tokens=32)

    assert len(profile["outliers_3x"]) == 0
    assert len(profile["outliers_5x"]) == 0
