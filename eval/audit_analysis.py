#!/usr/bin/env python3
"""
AuditKIT-style error analysis for the Lexsi Track 3 entry (in-window).

Reads the gold eval split and one predictions file and reports, beyond the
four headline metrics (produced by evaluate.py):
  - per-reason task success (already in results, re-shown with counts)
  - the silent-retry class: bank_server_error cases where the model
    contacted the customer (send_customer_message) instead of staying
    silent — the worst failure mode in this domain
  - invalid-call breakdown by problem type
  - most common argument errors (wrong value, by argument name)
  - failure examples (first few per reason)

Usage:
    python3 audit_analysis.py --gold ../data/eval.jsonl \
        --pred predictions_tuned.jsonl [--out audit_tuned.json]
Only numbers from actual runs are reported; nothing is invented.
"""

import argparse
import collections
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import validate_call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gold = {}
    tool_schemas = None
    with open(args.gold) as f:
        for line in f:
            row = json.loads(line)
            gold[row["id"]] = row
            if tool_schemas is None:
                tool_schemas = {t["name"]: t for t in row["tools"]}
    preds = {}
    with open(args.pred) as f:
        for line in f:
            row = json.loads(line)
            preds[row["id"]] = row

    missing = set(gold) - set(preds)
    if missing:
        raise SystemExit(f"predictions missing for {len(missing)} cases")

    silent = {"cases": 0, "silent_ok": 0, "contacted_customer": 0,
              "wrong_other": 0}
    invalid_kinds = collections.Counter()
    arg_errors = collections.Counter()
    failures = collections.defaultdict(list)
    parse_fail = 0

    for cid, row in gold.items():
        reason = row["scenario"]["failure_reason"]
        pred = preds[cid]
        if not pred.get("parse_ok"):
            parse_fail += 1
        expected = row["expected_calls"]
        predicted = pred["predicted_calls"]
        case_ok = (
            len(predicted) == len(expected)
            and all(predicted[i].get("tool") == e["tool"]
                    and predicted[i].get("arguments") == e["arguments"]
                    for i, e in enumerate(expected))
            and not any(validate_call(c, tool_schemas) for c in predicted)
        )
        if not case_ok and len(failures[reason]) < 3:
            failures[reason].append({
                "id": cid, "expected": expected, "predicted": predicted,
                "raw_output": pred.get("raw_output", "")[:400],
            })
        if reason == "bank_server_error":
            silent["cases"] += 1
            tools = [c.get("tool") for c in predicted]
            if tools == ["silent_retry"]:
                silent["silent_ok"] += 1
            elif "send_customer_message" in tools:
                silent["contacted_customer"] += 1
            else:
                silent["wrong_other"] += 1
        for c in predicted:
            for p in validate_call(c, tool_schemas):
                kind = p.split(":")[0]
                invalid_kinds[kind] += 1
        for i, exp in enumerate(expected):
            if i < len(predicted) and predicted[i].get("tool") == exp["tool"]:
                for k, v in exp["arguments"].items():
                    if predicted[i].get("arguments", {}).get(k) != v:
                        arg_errors[k] += 1

    report = {
        "pred_file": args.pred,
        "cases": len(gold),
        "parse_failures": parse_fail,
        "silent_retry_class_bank_server_error": silent,
        "invalid_call_problems": dict(invalid_kinds),
        "argument_errors_by_name": dict(arg_errors),
        "failure_examples": {k: v for k, v in failures.items()},
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
