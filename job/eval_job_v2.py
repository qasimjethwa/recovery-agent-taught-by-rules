#!/usr/bin/env python3
"""
Post-tuning evaluation job — Lexsi Track 3, in-window (Nov 8, 2026).

V2: adds --skip-baseline / --skip-tuned flags and writes to
/bucket/eval_v2/ (v1 outputs in /bucket/eval/ are untouched). Uses
/data/evaluate.py — upload evaluate_v2.py as evaluate.py for the v2
scorer (adds the structural task-success diagnostic; exact metric
unchanged).

Runs the AuditKIT-side audit on the held-out eval split for BOTH:
  1. the untuned base model (Aya Expanse 8B, zero-shot) — the baseline
  2. the LoRA-tuned model (base + adapter from the Nov 3 training job)
using the exact prompt construction from train/train_lora.py and
baseline/run_baseline.py, greedy decoding, then scores both prediction
files with eval/evaluate.py (the same scorer, same split).

Runs inside an HF Job with volumes:
  /model   — CohereLabs/aya-expanse-8b (model volume, gated, read-only)
  /data    — qasimjethwa/recovery-copilot-toolcalls (dataset volume:
             eval.jsonl, evaluate.py, this script)
  /bucket  — qasimjethwa/lexsi-adapter-output (bucket volume, read-write:
             adapter at /bucket/aya-expanse-8b-recovery-lora,
             outputs written to /bucket/eval/)

Outputs (also printed to stdout so the job logs carry the metrics):
  /bucket/eval/predictions_baseline.jsonl
  /bucket/eval/predictions_tuned.jsonl
  /bucket/eval/results_baseline.json
  /bucket/eval/results_tuned.json
"""

import json
import os
import subprocess
import sys
import time

SYSTEM_PROMPT = (
    "You are a payment-recovery agent. Given a failed payment and its diagnosed "
    "failure reason, respond ONLY with a JSON list of tool calls that executes "
    "the correct recovery plan, in order. Each call is an object with 'tool' "
    "and 'arguments'. Do not add commentary."
)

MODEL_PATH = "/model"
ADAPTER_PATH = "/bucket/aya-expanse-8b-recovery-lora"
EVAL_PATH = "/data/eval.jsonl"
OUT_DIR = "/bucket/eval_v2"
EVALUATE_PY = "/data/evaluate.py"


def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--skip-tuned", action="store_true")
    return ap.parse_args()


def build_user(row):
    return (
        row["instruction"]
        + "\n\nAvailable tools:\n"
        + json.dumps(row["tools"], ensure_ascii=False)
    )


def extract_json_list(text):
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


def run_model(model, tokenizer, rows, out_path, tag):
    import torch

    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    todo = [r for r in rows if r["id"] not in done]
    print(f"[{tag}] eval cases: {len(rows)}, already done: {len(done)}, "
          f"to run: {len(todo)}", flush=True)
    t0 = time.time()
    with open(out_path, "a") as f:
        for i, row in enumerate(todo):
            ct0 = time.time()
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user(row)},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=256, do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
            raw = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True)
            parsed = extract_json_list(raw)
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
            if (i + 1) % 16 == 0 or i == len(todo) - 1:
                el = time.time() - t0
                print(f"[{tag}] {len(done) + i + 1}/{len(rows)} "
                      f"parse_ok={ok} calls={len(calls)} "
                      f"elapsed={el / 60:.1f}m", flush=True)
    print(f"[{tag}] done in {(time.time() - t0) / 60:.1f} minutes", flush=True)


def score(pred_path, results_path):
    r = subprocess.run(
        [sys.executable, EVALUATE_PY, "--gold", EVAL_PATH,
         "--pred", pred_path, "--out", results_path],
        capture_output=True, text=True)
    print(r.stdout, flush=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr, flush=True)
        raise SystemExit(f"evaluate.py failed for {pred_path}")


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = [json.loads(l) for l in open(EVAL_PATH)]
    print(f"eval cases: {len(rows)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa")
    model.eval()
    print("base model loaded", flush=True)

    if not args.skip_baseline:
        baseline_pred = os.path.join(OUT_DIR, "predictions_baseline.jsonl")
        run_model(model, tokenizer, rows, baseline_pred, "baseline")
        score(baseline_pred, os.path.join(OUT_DIR, "results_baseline.json"))

    if not args.skip_tuned:
        from peft import PeftModel
        tuned = PeftModel.from_pretrained(model, ADAPTER_PATH)
        tuned.eval()
        print("LoRA adapter loaded", flush=True)
        tuned_pred = os.path.join(OUT_DIR, "predictions_tuned.jsonl")
        run_model(tuned, tokenizer, rows, tuned_pred, "tuned")
        score(tuned_pred, os.path.join(OUT_DIR, "results_tuned.json"))

    print("EVAL JOB COMPLETE", flush=True)


if __name__ == "__main__":
    main()
