---
title: Recovery Agent, Taught by Rules
emoji: 🛡️
colorFrom: blue
colorTo: yellow
sdk: static
pinned: false
license: apache-2.0
---

# Recovery Agent, Taught by Rules — demo

Lexsi AI Safety & Alignment Hackathon, Track 3 (Agentic Fine-tuning).

This demo shows the entry's **recorded** evaluation outputs: for each of
the 128 held-out cases, the gold recovery plan from the deterministic
Recovery Copilot rule engine, the base Aya Expanse 8B zero-shot plan, and
the LoRA-tuned model's plan, with the audit verdict for each.

- Adapter: [qasimjethwa/aya-expanse-8b-recovery-lora](https://huggingface.co/qasimjethwa/aya-expanse-8b-recovery-lora)
- Dataset + harness: [qasimjethwa/recovery-copilot-toolcalls](https://huggingface.co/datasets/qasimjethwa/recovery-copilot-toolcalls)

Nothing here is generated live — every prediction shown is a recorded
output of the Nov 8, 2026 evaluation run, so the demo cannot hallucinate
a result.
