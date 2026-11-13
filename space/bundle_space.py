#!/usr/bin/env python3
"""Bundle the REAL eval outputs into the HF Space demo directory.

Copies eval.jsonl + the v2 predictions/results into lexsi-entry/space/
so app.py serves recorded outputs (never live-generated or invented).
Run after the v2 eval job completes and its files are downloaded to
eval/results-2026-11-08-v2/ (or pass the source dir as argv[1]).

Usage: python3 bundle_space.py [results_dir]
"""
import json
import os
import shutil
import sys

ENTRY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPACE = os.path.join(ENTRY, "space")
V1 = os.path.join(ENTRY, "eval", "results-2026-11-08")
V2 = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ENTRY, "eval", "results-2026-11-08-v2")

# Baseline comes from the v1 eval (same split, same scorer family);
# tuned comes from the v2 re-eval.
FILES = {
    "eval.jsonl": os.path.join(ENTRY, "data", "eval.jsonl"),
    "predictions_baseline.jsonl": os.path.join(V1, "predictions_baseline.jsonl"),
    "predictions_tuned.jsonl": os.path.join(V2, "predictions_tuned.jsonl"),
    "results_baseline.json": os.path.join(
        ENTRY, "eval", "results_baseline_v2scorer.json"),
    "results_tuned.json": os.path.join(V2, "results_tuned.json"),
}

for name, src in FILES.items():
    if not os.path.exists(src):
        raise SystemExit(f"missing: {src}")
    shutil.copy(src, os.path.join(SPACE, name))
    n = sum(1 for _ in open(src)) if name.endswith(".jsonl") else None
    print(f"bundled {name}" + (f" ({n} rows)" if n else ""))

# Sanity: every prediction id must exist in eval, and both results files
# must be for 128 cases.
ev = {json.loads(l)["id"] for l in open(os.path.join(SPACE, "eval.jsonl"))}
for p in ("predictions_baseline.jsonl", "predictions_tuned.jsonl"):
    ids = {json.loads(l)["id"] for l in open(os.path.join(SPACE, p))}
    assert ids == ev, f"{p}: id mismatch ({len(ids)} vs {len(ev)})"
for r in ("results_baseline.json", "results_tuned.json"):
    d = json.load(open(os.path.join(SPACE, r)))
    assert d["cases"] == 128, f"{r}: cases={d['cases']}"
    print(f"{r}: task_success exact={d['metrics']['task_success_rate']} "
          f"structural={d['metrics'].get('task_success_rate_structural')}")
print("SPACE BUNDLE OK")
