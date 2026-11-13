# Entry Blueprint — Lexsi AI Safety and Alignment Hackathon

**Builder:** Qasim Jethwa · **Track:** 3 — Agentic Fine-tuning
**Working title:** *Recovery Agent, Taught by Rules: fine-tuning Aya Expanse 8B for reliable tool use in payment recovery, with a deterministic teacher and a continuous audit.*

All event facts below were verified on hackathon.lexsi.ai on 2026-10-23. No metric in this entry is claimed before the run that produces it.

---

## The idea in one paragraph

Failed-payment recovery is a domain where an agent's tool calls have real consequences: message a customer about a bank's server error and you erode trust; retry at the wrong time and you burn the one retry that would have worked; offer the wrong channel and the payment dies. Qasim's shipped **Recovery Copilot** — a deterministic rule engine that classifies a payment failure and prescribes the exact recovery action, channel, delay and message — becomes the **teacher and verifier**. Its rules generate tool-calling tasks whose correct answers are known by construction. We fine-tune **Aya Expanse 8B** on those tasks (AgentTune step), then audit the tuned agent with the same engine's ground truth (AuditKIT step): before vs. after, on held-out cases the model never trained on.

## Why this fits the safety/alignment theme

Reliability *is* the safety property for tool-using agents. This entry measures and improves the four failure modes that matter — wrong tool, wrong arguments, invalid calls, and the worst one in this domain: **contacting the customer when the correct action is silence** (bank-server-error cases). The audit reports that class separately, per failure reason.

---

## Requirement-by-requirement map

| Hackathon requirement (verified) | How this entry satisfies it | Artifact |
|---|---|---|
| Track 3 — Agentic Fine-tuning | The work is post-training an agent for reliable tool use, evaluated on tool-call accuracy/reliability | This blueprint; `train/`; `eval/` |
| Chain track repos into a real pipeline — AgentTune (post-train the agent) + AuditKIT (evaluate tool-call accuracy); 2-repo minimum relaxed for Track 3, both used anyway | AgentTune-style LoRA SFT produces the tuned agent; the audit harness scores base vs. tuned models with AuditKIT's target metrics (tool-call accuracy & reliability) | `train/train_lora.py`, `train/config.yaml`; `eval/evaluate.py` |
| Dense open-weights Cohere Labs model | **Aya Expanse 8B** is the primary (and only) base model | `train/config.yaml` → `base_model` |
| Before-and-after metrics, more than one | Four metrics (below), measured for the base model and again after fine-tuning, on the same held-out eval split | `eval/results_*.json` (in-window runs) |
| All artifacts hosted on Hugging Face (models, adapters, logs, datasets) | Dataset repo + adapter repo + training logs published to HF in-window; repo IDs are marked placeholders in the config until created | `train/config.yaml` → `output` (IN-WINDOW) |
| Public demo / repository by the deadline | This repository, public on GitHub; a Hugging Face Space demo of the tuned agent is an in-window deliverable | This repo; HF Space (IN-WINDOW) |
| HF Article write-up: metrics chosen and why, approach and rationale for datasets/tasks/models, how the model is meaningfully better | Article drafted in-window from the real run logs; outline in §"Article plan" below | HF Article (IN-WINDOW) |
| Projects developed during the hackathon (Nov 1–21) | The experiments — official dataset generation run, baseline audit, training, post-tuning audit, demo, article — all execute Nov 1–14. This foundation (generator, harness, training script) is preparatory scaffolding written ahead of the window and is described as such, honestly, in the README status | README → Status |
| Submission deadline: **end of day, Nov 14, 2026** | All artifacts frozen and submitted by Nov 14 EOD; judging Nov 15–16; Demo Day Nov 22 if shortlisted (top 20 informed by Nov 16) | Submission form (in-window) |
| Registration closes **Nov 5, 2026 (AoE)** | Qasim registers on hackathon.lexsi.ai before Nov 5; application details are his own (name, personal email, ATLAS SkillTech University, GitHub/HF/LinkedIn profiles) | Registration (Qasim, in-window) |

## The measurement plan (before / after)

Same held-out eval split, same harness, two runs:

1. **Baseline (IN-WINDOW):** Aya Expanse 8B, zero-shot, prompted with the scenario + tool schemas → predictions → `eval/evaluate.py`.
2. **After (IN-WINDOW):** the LoRA-tuned model on the identical split → same scorer.

Metrics (definitions in `eval/evaluate.py`):

- **Tool-call exact-match accuracy** — expected calls reproduced exactly (tool + all arguments) at the same sequence position.
- **Argument accuracy** — expected argument values predicted correctly.
- **Invalid-call rate** — predicted calls violating the tool schemas (unknown tool, missing/extra args, wrong types, enum/range violations).
- **End-to-end task success** — cases whose full plan matches gold exactly with zero invalid calls. Reported overall **and per failure reason**, so the silent-retry class (bank_server_error) is visible on its own.

Reference point available now: the **oracle** (an independent re-implementation of the rule engine) scores 1.0000 / 1.0000 / 0.0000 / 1.0000 on the eval split — that validates the harness and the port, and is *not* a model result. A deliberate-corruption control scores 0.7188 task success, proving the audit catches exactly the errors it claims to catch (details in the README).

## Dataset design

- **Source of truth:** Recovery Copilot's `FAILURE_RULES` — 8 failure reasons, each with a prescribed action, channel (SMS / WhatsApp / Email / System), delay (1–240 min) and message template bank (3 variants per channel, selected deterministically by transaction-ID hash). Ported faithfully to Python (`data/generate_dataset.py`); the original source is vendored in `reference/` for auditability.
- **Tool layer (new, designed for this entry):** `schedule_retry`, `send_customer_message`, `offer_alternate_method`, `silent_retry`. Each rule's action text maps to one deterministic gold call sequence (1–3 calls per case).
- **Splits:** train = 400 cases, balanced 50 per reason. Eval = 128 held-out cases — a natural hash-driven mix plus 8 edge cases where UPI is the original method (flipping the alternate-method branch to Wallet) and the silent case. Transaction IDs are disjoint across splits (asserted in code; verified 0 overlap).
- **Why synthetic-but-verified beats scraped:** every gold label is correct *by construction* — the same determinism that makes the engine safe to run a merchant's recovery flow on makes its labels safe to train and audit against.

## Article plan (HF Article, in-window)

1. The problem: agent tool-use errors are payment errors — with the silent-failure case as the running example.
2. The teacher: how a shipped rule engine becomes a task generator and verifier (fidelity port, cross-implementation check).
3. Method: dataset design, LoRA SFT setup (AgentTune step), audit design (AuditKIT step).
4. Results: before/after table (4 metrics, overall + per-reason), error analysis from the audit logs — only numbers from actual runs.
5. Cost/ROI (preferred by the rules): self-hosted tuned 8B vs. closed-source token billing for a realistic recovery volume — computed in-window from real run costs.
6. Limits: synthetic distribution, single-domain, template-bound messages; what transfers and what doesn't.

## In-window schedule

| When (2026) | Step |
|---|---|
| By Nov 5 | Qasim registered (registration closes Nov 5 AoE) |
| Nov 1 | Window opens — regenerate the official dataset with the shipped generator; publish dataset to HF |
| Nov 1–2 | Baseline audit: base Aya Expanse 8B on the eval split (AuditKIT step) |
| Nov 2–6 | LoRA SFT runs (AgentTune step); adapter + logs to HF |
| Nov 6–9 | Post-tuning audit on the identical split; error analysis; iterate if a class (e.g. silent-retry) lags |
| Nov 9–12 | HF Space demo; ROI comparison; HF Article drafted from real logs |
| Nov 13 | Freeze artifacts; final end-to-end check of every link |
| **Nov 14 EOD** | **Submit** |
| Nov 15–16 | Judging; Nov 16–21 thesis-proof window if shortlisted; Nov 22 Demo Day |

## Honesty commitments

- Only shipped work is claimed: Recovery Copilot exists and does what its README says; this foundation's scripts run and their outputs are recorded verbatim.
- The engine's per-rule `successRate` values are its *configured assumptions* for the original demo's simulation — never presented as measured results.
- Baseline and tuned-model numbers exist only after their in-window runs, and will be reported whether or not they flatter the entry.
