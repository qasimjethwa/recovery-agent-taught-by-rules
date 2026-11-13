#!/usr/bin/env python3
"""
Evaluation harness for the Lexsi Track 3 entry (the AuditKIT-side scorer).

Scores predicted tool calls against the dataset's expected calls and reports
the four audit metrics the hackathon entry is built around:

  1. call_exact_match_accuracy — of all expected calls, the fraction reproduced
     exactly (same tool, same arguments) at the same position in the sequence.
  2. argument_accuracy — of all expected argument values (in position-aligned
     calls with the correct tool), the fraction predicted correctly.
  3. invalid_call_rate — of all predicted calls, the fraction that are invalid:
     unknown tool, missing/extra arguments, wrong types, enum violations, or
     out-of-range values, judged against the tool schemas in the dataset.
  4. task_success_rate — fraction of cases whose predicted sequence matches the
     expected sequence exactly, end to end, with no invalid calls.

Usage:
    python3 evaluate.py --gold ../data/eval.jsonl --pred predictions_oracle.jsonl --out results_oracle.json
"""

import argparse
import json

TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
}


def validate_call(call, tool_schemas):
    """Return a list of validity problems for one predicted call ([] = valid)."""
    problems = []
    name = call.get("tool")
    if name not in tool_schemas:
        return [f"unknown tool: {name!r}"]
    schema = tool_schemas[name]
    params = schema["parameters"]
    args = call.get("arguments")
    if not isinstance(args, dict):
        return ["arguments is not an object"]
    for req in schema["required"]:
        if req not in args:
            problems.append(f"missing required argument: {req}")
    for key, val in args.items():
        if key not in params:
            problems.append(f"unexpected argument: {key}")
            continue
        spec = params[key]
        if not TYPE_CHECKS[spec["type"]](val):
            problems.append(f"argument {key}: expected {spec['type']}")
            continue
        if "enum" in spec and val not in spec["enum"]:
            problems.append(f"argument {key}: {val!r} not in enum")
        if "minimum" in spec and isinstance(val, int) and val < spec["minimum"]:
            problems.append(f"argument {key}: below minimum {spec['minimum']}")
    return problems


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
            preds[row["id"]] = row["predicted_calls"]

    missing = set(gold) - set(preds)
    if missing:
        raise SystemExit(f"predictions missing for {len(missing)} cases, e.g. {sorted(missing)[:3]}")

    n_cases = len(gold)
    total_expected_calls = 0
    exact_calls = 0
    total_expected_args = 0
    correct_args = 0
    total_pred_calls = 0
    invalid_calls = 0
    successful_cases = 0
    per_reason = {}

    for cid, row in gold.items():
        expected = row["expected_calls"]
        predicted = preds[cid]
        stats = per_reason.setdefault(row["scenario"]["failure_reason"],
                                      {"cases": 0, "success": 0})
        stats["cases"] += 1

        pred_validity = [validate_call(c, tool_schemas) for c in predicted]
        total_pred_calls += len(predicted)
        invalid_calls += sum(1 for p in pred_validity if p)

        total_expected_calls += len(expected)
        for i, exp in enumerate(expected):
            if i < len(predicted) and predicted[i].get("tool") == exp["tool"]:
                if predicted[i].get("arguments") == exp["arguments"]:
                    exact_calls += 1
                for k, v in exp["arguments"].items():
                    total_expected_args += 1
                    if predicted[i].get("arguments", {}).get(k) == v:
                        correct_args += 1
            else:
                total_expected_args += len(exp["arguments"])

        case_ok = (
            len(predicted) == len(expected)
            and all(predicted[i].get("tool") == e["tool"]
                    and predicted[i].get("arguments") == e["arguments"]
                    for i, e in enumerate(expected))
            and not any(pred_validity)
        )
        if case_ok:
            successful_cases += 1
            stats["success"] += 1

    results = {
        "gold_file": args.gold,
        "pred_file": args.pred,
        "cases": n_cases,
        "metrics": {
            "call_exact_match_accuracy": round(exact_calls / total_expected_calls, 4) if total_expected_calls else 0.0,
            "argument_accuracy": round(correct_args / total_expected_args, 4) if total_expected_args else 0.0,
            "invalid_call_rate": round(invalid_calls / total_pred_calls, 4) if total_pred_calls else 0.0,
            "task_success_rate": round(successful_cases / n_cases, 4) if n_cases else 0.0,
        },
        "counts": {
            "expected_calls": total_expected_calls,
            "exact_calls": exact_calls,
            "expected_argument_values": total_expected_args,
            "correct_argument_values": correct_args,
            "predicted_calls": total_pred_calls,
            "invalid_calls": invalid_calls,
            "successful_cases": successful_cases,
        },
        "per_reason_task_success": {
            k: {"cases": v["cases"], "successful": v["success"],
                "rate": round(v["success"] / v["cases"], 4)}
            for k, v in sorted(per_reason.items())
        },
    }
    print(json.dumps(results, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
