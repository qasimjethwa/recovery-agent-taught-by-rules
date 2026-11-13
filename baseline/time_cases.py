#!/usr/bin/env python3
"""Steady-state timing: run N eval cases back-to-back in one process and
time each case separately (load time is one-off; per-case time after the
first case is the steady-state cost that decides full-run feasibility)."""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ENTRY = os.path.dirname(HERE)
MODEL = os.path.join(ENTRY, "models", "aya-expanse-8b-Q4_K_M.gguf")
SYSTEM_PROMPT = (
    "You are a payment-recovery agent. Given a failed payment and its diagnosed "
    "failure reason, respond ONLY with a JSON list of tool calls that executes "
    "the correct recovery plan, in order. Each call is an object with 'tool' "
    "and 'arguments'. Do not add commentary."
)

n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
rows = [json.loads(l) for l in open(os.path.join(ENTRY, "data", "eval.jsonl"))][:n]

from llama_cpp import Llama

t0 = time.time()
llm = Llama(model_path=MODEL, n_ctx=1024, n_threads=4, verbose=False)
print(f"loaded in {time.time()-t0:.1f}s", flush=True)
for row in rows:
    user = row["instruction"] + "\n\nAvailable tools:\n" + json.dumps(
        row["tools"], ensure_ascii=False)
    ct0 = time.time()
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=200,
    )
    dt = time.time() - ct0
    u = out.get("usage", {})
    ct = u.get("completion_tokens", 0)
    print(f"{row['id']} gold={row['scenario']['failure_reason']} "
          f"gen_tokens={ct} seconds={dt:.1f} "
          f"sec/token={dt/ct if ct else 0:.2f}", flush=True)
    print("  RAW:", repr(out["choices"][0]["message"]["content"][:300]), flush=True)
