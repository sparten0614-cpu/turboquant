#!/usr/bin/env python3
"""Quick Needle-In-A-Haystack test for KV cache quality validation."""

import argparse, json, random, time
from openai import OpenAI

FILLER = """The history of maritime navigation spans thousands of years, from the ancient Polynesian wayfinders who read stars and ocean swells, to the development of magnetic compasses in medieval China, to the GPS satellites orbiting Earth today. Each advance represented not just a technological leap but a fundamental shift in how humans understood their relationship to geography and distance. The chronometer, invented by John Harrison in the 18th century, solved the longitude problem that had plagued sailors for centuries, enabling accurate east-west positioning for the first time. """

NEEDLE = "The secret code for this test is: PURPLE-ELEPHANT-7492."
QUESTION = "What is the secret code mentioned in the text?"
EXPECTED = "PURPLE-ELEPHANT-7492"

def build_context(target_tokens, needle_depth=0.5):
    """Build context with needle inserted at specified depth."""
    # Rough estimate: 1 token ≈ 4 chars
    target_chars = target_tokens * 4
    filler_needed = target_chars - len(NEEDLE)

    repeats = (filler_needed // len(FILLER)) + 1
    full_filler = FILLER * repeats

    insert_pos = int(len(full_filler[:filler_needed]) * needle_depth)
    context = full_filler[:insert_pos] + "\n\n" + NEEDLE + "\n\n" + full_filler[insert_pos:filler_needed]
    return context

def test_niah(client, model, context_tokens, depths, runs_per_depth=1):
    """Run NIAH test at various depths."""
    results = []
    for depth in depths:
        for run in range(runs_per_depth):
            context = build_context(context_tokens, depth)
            prompt = f"Read the following text carefully:\n\n{context}\n\nQuestion: {QUESTION}\nAnswer concisely with just the code."

            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=64,
                )
                answer = resp.choices[0].message.content.strip()
                found = EXPECTED in answer
                results.append({
                    "depth": depth,
                    "found": found,
                    "answer": answer[:200],
                    "context_tokens": context_tokens,
                })
                status = "PASS" if found else "FAIL"
                print(f"  depth={depth:.1%} {status} | answer: {answer[:80]}")
            except Exception as e:
                print(f"  depth={depth:.1%} ERROR: {e}")
                results.append({"depth": depth, "found": False, "answer": str(e), "context_tokens": context_tokens})
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", "-t", required=True, help="Run tag (e.g. f16, tqkv6)")
    parser.add_argument("--port", "-p", type=int, default=8080)
    parser.add_argument("--context", "-c", type=int, default=4096, help="Context length in tokens")
    args = parser.parse_args()

    client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="token-abc123")

    # Test at 5 depths
    depths = [0.1, 0.3, 0.5, 0.7, 0.9]

    print(f"NIAH Test [{args.tag}] — {args.context} tokens, {len(depths)} depths")
    print("=" * 60)

    results = test_niah(client, "model", args.context, depths)

    passed = sum(1 for r in results if r["found"])
    total = len(results)

    print(f"\n{'=' * 60}")
    print(f"Tag: {args.tag}")
    print(f"Context: {args.context} tokens")
    print(f"Result: {passed}/{total} = {passed/total*100:.0f}%")
    print(f"{'=' * 60}")

    # Save
    import os
    os.makedirs("results", exist_ok=True)
    with open(f"results/niah_{args.tag}.json", "w") as f:
        json.dump({"tag": args.tag, "context": args.context, "passed": passed, "total": total,
                    "accuracy": passed/total*100, "details": results}, f, indent=2)

if __name__ == "__main__":
    main()
