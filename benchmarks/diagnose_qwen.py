"""
Diagnose why Qwen2.5-3B gives PPL 350 with TurboQuant.

Possible causes:
1. Large K norms → large absolute errors → softmax breakdown
2. fp16 x_norm precision loss
3. Codebook mismatch (unit-sphere distribution differs from N(0,1/d))
4. Per-channel outliers surviving rotation
"""

import sys
import os
import torch
import numpy as np
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from turboquant.turboquant import TurboQuantConfig, TurboQuantCompressor
from turboquant.codebook import quantize_scalar, dequantize_scalar


def analyze_kv_norms(model_name, device="mps", max_tokens=256):
    """Analyze KV cache vector norms and per-channel statistics."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device,
    )
    model.eval()
    head_dim = getattr(model.config, 'head_dim',
                       model.config.hidden_size // model.config.num_attention_heads)

    # Get some text
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=max_tokens)["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        cache = outputs.past_key_values

    print(f"\nModel: {model_name}")
    print(f"head_dim={head_dim}, KV heads={model.config.num_key_value_heads}, "
          f"layers={model.config.num_hidden_layers}")
    print(f"Tokens: {input_ids.shape[1]}")
    print()

    # Analyze K norms across all layers
    all_k_norms = []
    all_channel_maxabs = []

    for layer_idx, layer in enumerate(cache.layers):
        if not layer.is_initialized:
            continue
        keys = layer.keys[0].float().cpu().numpy()  # (num_heads, seq, head_dim)
        norms = np.linalg.norm(keys, axis=-1)  # (num_heads, seq)
        all_k_norms.append(norms.flatten())

        # Per-channel max abs across sequence
        channel_max = np.max(np.abs(keys), axis=1)  # (num_heads, head_dim)
        all_channel_maxabs.append(channel_max)

    all_k_norms = np.concatenate(all_k_norms)
    all_channel_maxabs = np.stack(all_channel_maxabs)  # (layers, heads, head_dim)

    print("=== K Vector Norms ===")
    print(f"  min: {np.min(all_k_norms):.2f}")
    print(f"  mean: {np.mean(all_k_norms):.2f}")
    print(f"  max: {np.max(all_k_norms):.2f}")
    print(f"  p50: {np.percentile(all_k_norms, 50):.2f}")
    print(f"  p90: {np.percentile(all_k_norms, 90):.2f}")
    print(f"  p99: {np.percentile(all_k_norms, 99):.2f}")

    print("\n=== Per-Channel Max Abs (across all layers/heads) ===")
    channel_max_flat = all_channel_maxabs.reshape(-1, head_dim)
    overall_channel_max = np.max(channel_max_flat, axis=0)  # (head_dim,)
    print(f"  Global max per channel: max={np.max(overall_channel_max):.2f}, "
          f"min={np.min(overall_channel_max):.2f}, "
          f"mean={np.mean(overall_channel_max):.2f}")

    # Outlier analysis: how many channels have max >> mean?
    mean_max = np.mean(overall_channel_max)
    outlier_channels = np.sum(overall_channel_max > 3 * mean_max)
    print(f"  Channels with max > 3x mean: {outlier_channels}/{head_dim}")

    # Compression quality analysis
    print("\n=== Compression Quality Analysis ===")
    config = TurboQuantConfig(head_dim=head_dim, total_bits=4)
    comp = TurboQuantCompressor(config)

    # Sample vectors from the first layer
    first_layer_keys = cache.layers[0].keys[0, 0].float().cpu().numpy()  # (seq, head_dim)

    cos_sims = []
    abs_errors = []
    rel_errors = []
    norm_ratios = []

    for s in range(min(first_layer_keys.shape[0], 128)):
        x = first_layer_keys[s].astype(np.float64)
        x_norm = np.linalg.norm(x)
        compressed = comp.compress(x)
        x_hat = comp.decompress(compressed)

        cos = np.dot(x, x_hat) / (np.linalg.norm(x) * np.linalg.norm(x_hat) + 1e-10)
        cos_sims.append(cos)
        abs_errors.append(np.linalg.norm(x - x_hat))
        rel_errors.append(np.linalg.norm(x - x_hat) / (x_norm + 1e-10))
        norm_ratios.append(np.linalg.norm(x_hat) / (x_norm + 1e-10))

    print(f"  4-bit cos_sim: mean={np.mean(cos_sims):.4f}, min={np.min(cos_sims):.4f}")
    print(f"  abs error: mean={np.mean(abs_errors):.4f}, max={np.max(abs_errors):.4f}")
    print(f"  rel error: mean={np.mean(rel_errors):.4f}, max={np.max(rel_errors):.4f}")
    print(f"  norm ratio (hat/orig): mean={np.mean(norm_ratios):.4f}")

    # Check if fp16 norm storage is lossy
    print("\n=== fp16 Norm Precision ===")
    norms_fp32 = np.array([np.linalg.norm(first_layer_keys[s]) for s in range(first_layer_keys.shape[0])])
    norms_fp16 = norms_fp32.astype(np.float16).astype(np.float64)
    norm_error = np.abs(norms_fp32 - norms_fp16) / (norms_fp32 + 1e-10)
    print(f"  fp16 norm relative error: mean={np.mean(norm_error):.6f}, "
          f"max={np.max(norm_error):.6f}")

    # Attention logit error analysis
    print("\n=== Attention Logit Error Analysis ===")
    # Sample a query vector and compute Q·K errors
    q = np.random.default_rng(42).standard_normal(head_dim)
    q = q / np.linalg.norm(q) * np.mean(norms_fp32)  # Scale Q to typical magnitude

    logit_errors = []
    for s in range(min(first_layer_keys.shape[0], 128)):
        x = first_layer_keys[s].astype(np.float64)
        compressed = comp.compress(x)
        x_hat = comp.decompress(compressed)

        true_logit = np.dot(q, x) / math.sqrt(head_dim)
        est_logit = np.dot(q, x_hat) / math.sqrt(head_dim)
        logit_errors.append(abs(true_logit - est_logit))

    print(f"  Q·K / sqrt(d) error: mean={np.mean(logit_errors):.4f}, "
          f"max={np.max(logit_errors):.4f}")
    print(f"  For softmax, error > ~2 is catastrophic (changes probabilities 10x)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--tokens", type=int, default=256)
    args = parser.parse_args()
    analyze_kv_norms(args.model, args.device, args.tokens)
