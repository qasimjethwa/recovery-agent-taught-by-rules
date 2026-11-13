#!/usr/bin/env python3
"""Smoke test: load the Q4_K_M GGUF and generate a few tokens for eval-0001.

Prints load time, prompt token count, generation time and the raw output.
Bounded by the caller's `timeout`; if this cannot finish a handful of
tokens quickly, a 128-case CPU baseline on this VM is not feasible.
"""
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

row = json.loads(open(os.path.join(ENTRY, "data", "eval.jsonl")).readline())
user = row["instruction"] + "\n\nAvailable tools:\n" + json.dumps(
    row["tools"], ensure_ascii=False)

from llama_cpp import Llama

t0 = time.time()
llm = Llama(model_path=MODEL, n_ctx=1024, n_threads=4, verbose=False)
print(f"loaded in {time.time()-t0:.1f}s", flush=True)
toks = llm.tokenize(user.encode("utf-8"))
print(f"user prompt tokens: {len(toks)}", flush=True)

t0 = time.time()
out = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ],
    temperature=0.0,
    max_tokens=16,
)
dt = time.time() - t0
print(f"16 tokens in {dt:.1f}s", flush=True)
print("RAW:", repr(out["choices"][0]["message"]["content"]), flush=True)
print("usage:", out.get("usage"), flush=True)
