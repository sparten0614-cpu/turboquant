"""
Perplexity benchmark for TurboQuant KV cache compression.

Correct approach: populate KV cache with a normal forward pass (prefill),
then compress the cache, then evaluate on subsequent tokens using the
compressed cache. This matches real-world deployment where old KV entries
are compressed and new tokens use the compressed cache for attention.

Two modes:
  1. "prefill-then-compress": prefill N context tokens normally, compress
     the KV cache, then evaluate perplexity on remaining tokens.
  2. "compress-during-prefill" (legacy): compress during the forward pass
     itself. Included for comparison but NOT recommended — errors compound
     through layers.
"""

import sys
import os
import time
import math
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from turboquant.hf_integration import TurboQuantCache, TurboQuantCacheLayer
from turboquant.turboquant import TurboQuantConfig, TurboQuantCompressor


def load_model_and_tokenizer(model_name: str, device: str = "mps"):
    """Load model in fp16 on the specified device."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    model.config.head_dim = head_dim
    print(f"  Loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    print(f"  head_dim={head_dim}, "
          f"num_kv_heads={model.config.num_key_value_heads}, "
          f"num_layers={model.config.num_hidden_layers}")
    return model, tokenizer


def load_eval_data(tokenizer, max_tokens: int = 512):
    """Load WikiText-2 validation data for perplexity evaluation."""
    from datasets import load_dataset

    print("Loading WikiText-2 validation set...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    encodings = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=max_tokens)
    input_ids = encodings["input_ids"]
    print(f"  Tokens: {input_ids.shape[1]}")
    return input_ids


def compress_dynamic_cache(dynamic_cache, total_bits: int, head_dim: int,
                           per_channel_scaling: bool = False) -> TurboQuantCache:
    """Compress a HF DynamicCache into a TurboQuantCache.

    Extracts K,V tensors from each layer and compresses them.
    """
    config = TurboQuantConfig(head_dim=head_dim, total_bits=total_bits)
    compressor = TurboQuantCompressor(config)

    tq_cache = TurboQuantCache(total_bits=total_bits, head_dim=head_dim,
                               per_channel_scaling=per_channel_scaling)

    for layer_idx, layer in enumerate(dynamic_cache.layers):
        if not layer.is_initialized:
            continue
        keys = layer.keys    # (batch, num_kv_heads, seq_len, head_dim)
        values = layer.values

        num_heads = keys.shape[1]
        tq_layer = TurboQuantCacheLayer(compressor, num_heads,
                                        per_channel_scaling=per_channel_scaling)
        tq_layer.is_initialized = True

        k_np = keys[0].detach().cpu().float().numpy().astype(np.float64)
        v_np = values[0].detach().cpu().float().numpy().astype(np.float64)

        # Compute per-channel scales from the full prefill sequence
        if per_channel_scaling:
            tq_layer._k_scales = tq_layer._compute_channel_scales(k_np)
            tq_layer._v_scales = tq_layer._compute_channel_scales(v_np)

        for h in range(num_heads):
            for s in range(k_np.shape[1]):
                k_vec = k_np[h, s]
                v_vec = v_np[h, s]
                if per_channel_scaling:
                    k_vec = k_vec / tq_layer._k_scales[h]
                    v_vec = v_vec / tq_layer._v_scales[h]
                tq_layer._keys[h].append(compressor.compress(k_vec))
                tq_layer._values[h].append(compressor.compress(v_vec))

        tq_layer._seen_tokens = k_np.shape[1]
        tq_cache.layers.append(tq_layer)

    tq_cache._seen_tokens = tq_cache.layers[0].seq_length if tq_cache.layers else 0
    return tq_cache


def eval_prefill_then_compress(model, input_ids, total_bits: int, context_len: int,
                                head_dim: int, label: str = "",
                                per_channel_scaling: bool = False):
    """Correct evaluation: prefill normally, compress cache, evaluate on remaining tokens.

    Args:
        model: HF model
        input_ids: full evaluation sequence (1, total_len)
        total_bits: TurboQuant bit level
        context_len: number of tokens to prefill normally
        head_dim: model head dimension

    Returns:
        (perplexity, elapsed_seconds, tq_cache)
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    total_len = input_ids.shape[1]
    eval_len = total_len - context_len

    if eval_len < 2:
        raise ValueError(f"Not enough eval tokens: total={total_len}, context={context_len}")

    print(f"\n[{label}] Prefill {context_len} tokens, eval {eval_len} tokens...")
    t0 = time.time()

    with torch.no_grad():
        # Step 1: Prefill with normal cache
        context_ids = input_ids[:, :context_len]
        outputs = model(context_ids, use_cache=True)
        dynamic_cache = outputs.past_key_values

        # Step 2: Compress the cache
        t_compress = time.time()
        if total_bits > 0:
            cache = compress_dynamic_cache(dynamic_cache, total_bits, head_dim,
                                           per_channel_scaling=per_channel_scaling)
        else:
            cache = dynamic_cache  # baseline: keep uncompressed
        compress_time = time.time() - t_compress

        # Step 3: Evaluate remaining tokens one at a time
        total_loss = 0.0
        num_tokens = 0

        for i in range(context_len, total_len):
            token_id = input_ids[:, i:i+1]
            prev_token = input_ids[:, i-1:i]

            outputs = model(prev_token, past_key_values=cache, use_cache=True)
            cache = outputs.past_key_values

            # Get logit for actual next token
            logits = outputs.logits[:, -1, :]  # (1, vocab)
            loss = torch.nn.functional.cross_entropy(logits, token_id.squeeze(0))
            total_loss += loss.item()
            num_tokens += 1

    elapsed = time.time() - t0
    avg_loss = total_loss / num_tokens
    ppl = math.exp(avg_loss)

    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {ppl:.2f}")
    print(f"  Time: {elapsed:.1f}s (compress: {compress_time:.1f}s)")

    if total_bits > 0 and hasattr(cache, "memory_usage_bytes"):
        # The cache has grown during eval; report the compressed portion
        pass

    return ppl, elapsed


def main():
    import argparse
    parser = argparse.ArgumentParser(description="TurboQuant Perplexity Benchmark")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="HuggingFace model name")
    parser.add_argument("--tokens", type=int, default=512,
                        help="Total tokens (context + eval)")
    parser.add_argument("--context", type=int, default=256,
                        help="Context tokens (prefilled normally, then compressed)")
    parser.add_argument("--device", default="mps",
                        help="Device (mps, cuda, cpu)")
    parser.add_argument("--bits", nargs="+", type=int, default=[2, 3, 4],
                        help="TurboQuant bit levels to test")
    parser.add_argument("--per-channel-scaling", action="store_true",
                        help="Enable per-channel scaling (for models with large K norms)")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    input_ids = load_eval_data(tokenizer, max_tokens=args.tokens)

    head_dim = model.config.head_dim
    context_len = min(args.context, input_ids.shape[1] - 32)

    # Baseline: prefill + eval with normal DynamicCache
    ppl_baseline, t_baseline = eval_prefill_then_compress(
        model, input_ids, total_bits=0, context_len=context_len,
        head_dim=head_dim, label="Baseline (fp16 cache)")

    results = [("Baseline (fp16)", ppl_baseline, t_baseline, "-", "-")]

    # TurboQuant at each bit level
    for bits in args.bits:
        label = f"TurboQuant-{bits}bit"
        if args.per_channel_scaling:
            label += "+pcs"
        ppl, elapsed = eval_prefill_then_compress(
            model, input_ids, total_bits=bits, context_len=context_len,
            head_dim=head_dim, label=label,
            per_channel_scaling=args.per_channel_scaling)

        delta_ppl = ((ppl - ppl_baseline) / ppl_baseline) * 100
        results.append((label, ppl, elapsed, "", ""))
        print(f"  PPL delta: {delta_ppl:+.2f}%")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY — Prefill-then-Compress Evaluation")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Context: {context_len} tokens (prefilled normally)")
    print(f"Eval: {input_ids.shape[1] - context_len} tokens (using compressed cache)")
    print(f"Head dim: {head_dim}, KV heads: {model.config.num_key_value_heads}, "
          f"Layers: {model.config.num_hidden_layers}")
    print()
    print(f"{'Method':<25} {'PPL':>8} {'Δ PPL':>10} {'Time':>8}")
    print("-" * 55)
    for name, ppl, elapsed, _, _ in results:
        if name.startswith("Baseline"):
            print(f"{name:<25} {ppl:>8.2f} {'—':>10} {elapsed:>7.1f}s")
        else:
            delta = ((ppl - ppl_baseline) / ppl_baseline) * 100
            print(f"{name:<25} {ppl:>8.2f} {delta:>+9.2f}% {elapsed:>7.1f}s")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
