#!/usr/bin/env python3
"""
Baseline audit runner — Lexsi Track 3, in-window run (Nov 1, 2026).

Evaluates the UNTUNED Aya Expanse 8B (bartowski Q4_K_M GGUF quant of
CohereLabs/aya-expanse-8b — the official weights are gated and no HF token
was available on this VM, so the full-precision weights could not be
downloaded; the quantisation is disclosed in the run log and results file)
zero-shot on the held-out eval split, using the exact prompt construction
from train/train_lora.py (same system prompt, same user = instruction +
tool schemas) so the baseline is directly comparable to the tuned run.

Greedy decoding (temperature 0). Output: predictions_baseline.jsonl with
one row per eval case: id, predicted_calls (parsed JSON list, [] when the
model produced no parseable call list), raw output, and parse status.
Scoring is done separately with eval/evaluate.py — never hand-computed.

Usage:
    ../../.venv/bin/python run_baseline.py [--limit N] [--model PATH]
Resume-safe: cases already present in the output file are skipped.
"""

import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ENTRY = os.path.dirname(HERE)

SYSTEM_PROMPT = (
    "You are a payment-recovery agent. Given a failed payment and its diagnosed "
    "failure reason, respond ONLY with a JSON list of tool calls that executes "
    "the correct recovery plan, in order. Each call is an object with 'tool' "
    "and 'arguments'. Do not add commentary."
)


def build_user(row):
    return (
        row["instruction"]
        + "\n\nAvailable tools:\n"
        + json.dumps(row["tools"], ensure_ascii=False)
    )


def extract_json_list(text):
    """Return the first balanced top-level JSON array in text, or None."""
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        val = json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        return None
                    return val if isinstance(val, list) else None
    return None


def normalise_calls(val):
    """Keep only well-formed {tool, arguments-dict} calls; else []."""
    if not isinstance(val, list):
        return [], False
    calls = []
    for c in val:
        if isinstance(c, dict) and isinstance(c.get("tool"), str) \
                and isinstance(c.get("arguments"), dict):
            calls.append({"tool": c["tool"], "arguments": c["arguments"]})
        else:
            return [], False
    return calls, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(
        ENTRY, "models", "aya-expanse-8b-Q4_K_M.gguf"))
    ap.add_argument("--eval", default=os.path.join(ENTRY, "data", "eval.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "predictions_baseline.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    from llama_cpp import Llama

    rows = [json.loads(l) for l in open(args.eval)]
    if args.limit:
        rows = rows[:args.limit]

    done = {}
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
                done[r["id"]] = True
            except (json.JSONDecodeError, KeyError):
                continue
    todo = [r for r in rows if r["id"] not in done]
    print(f"eval cases: {len(rows)}, already done: {len(done)}, to run: {len(todo)}",
          flush=True)
    if not todo:
        return

    llm = Llama(model_path=args.model, n_ctx=args.n_ctx, n_threads=args.threads,
                verbose=False)
    n_tok = len(llm.tokenize(build_user(rows[0]).encode("utf-8")))
    print(f"sample prompt tokens (first case, user turn only): ~{n_tok}", flush=True)

    t0 = time.time()
    with open(args.out, "a") as f:
        for i, row in enumerate(todo):
            ct0 = time.time()
            out = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user(row)},
                ],
                temperature=0.0,
                max_tokens=320,
                stop=["<|endoftext|>", "<|im_end|>", "]"],
            )
            raw = out["choices"][0]["message"]["content"]
            parsed = extract_json_list(raw)
            if parsed is None and raw.count("[") > raw.count("]"):
                # Generation was stopped at the list's closing bracket
                # (stop strings are excluded from the returned text).
                parsed = extract_json_list(raw + "]")
            calls, ok = normalise_calls(parsed) if parsed is not None else ([], False)
            rec = {
                "id": row["id"],
                "predicted_calls": calls,
                "raw_output": raw,
                "parse_ok": ok,
                "seconds": round(time.time() - ct0, 1),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_done = len(done) + i + 1
            if n_done % 10 == 0 or i == len(todo) - 1:
                el = time.time() - t0
                rate = (i + 1) / el if el > 0 else 0
                print(f"[{n_done}/{len(rows)}] {row['id']} "
                      f"parse_ok={ok} calls={len(calls)} "
                      f"elapsed={el/60:.1f}m rate={rate*60:.1f} cases/min",
                      flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
