# Recovery Agent, Taught by Rules: fine-tuning Aya Expanse 8B for reliable tool use in payment recovery

*Lexsi AI Safety & Alignment Hackathon — Track 3: Agentic Fine-tuning. By Qasim Jethwa.*

## The problem: agent tool-use errors are payment errors

A payment-recovery agent acts on real money and real customer trust. Its
tool calls have consequences a chatbot never faces: message a customer
about a bank-side server error and you erode trust for nothing; retry at
the wrong time and you burn the one retry that would have worked; offer
the wrong channel and the payment dies. The worst failure in this domain
isn't a wrong answer — it's contacting the customer when the correct
action is silence.

That is the safety property this entry measures and improves: can a
fine-tuned agent execute a recovery plan with the right tools, in the
right order, with the right arguments — and stay silent when silence is
the plan?

## The teacher: a rule engine that already ships

Qasim's [Recovery Copilot](https://github.com/qasimjethwa/Failed-payment-recovery)
is a deterministic rule engine that classifies a payment failure and
prescribes the exact recovery action, channel, delay and message. It
ships as a working product, and its determinism is what makes it a
teacher: its rules generate tool-calling tasks whose correct answers are
known **by construction**, and the same engine grades any agent against
that ground truth. The engine is both the training-data generator and
the verifier — the AgentTune step and the AuditKIT step of one pipeline.

The engine's 8 failure reasons (insufficient funds, OTP timeout, bank
server error, card declined, network drop, incorrect CVV, expired card,
daily limit exceeded) map to 4 tools — `schedule_retry`,
`send_customer_message`, `offer_alternate_method`, `silent_retry` — with
a deterministic gold call sequence (1–3 calls) per case. The dataset is
400 training cases (50 per reason) and a held-out eval split of 128
cases with disjoint transaction IDs (0 overlap), including 8 edge cases
where UPI is the original method and the bank-error silence rule is
exercised. Synthetic — but verified: an independent re-implementation of
the engine (the oracle) agrees with the generator on all 243 eval calls,
and a deliberately corrupted control run drops to 0% success in exactly
the two classes it corrupts, proving the audit catches the errors it
claims to catch.

## Method

**Model:** CohereLabs/aya-expanse-8b, a dense open-weights 8B model —
the only base model in this entry.

**AgentTune step (post-training):** LoRA SFT (r=16, alpha=32,
dropout=0.05, q/k/v/o/gate/up/down projections), completion-only loss,
3 epochs, lr 1e-4 cosine, bf16, on 1× NVIDIA L40S (HF Jobs). 75 steps,
423 s running, train_loss 0.1653, mean_token_accuracy 0.9946.

**AuditKIT step (evaluation):** the same harness, the same held-out
split, two runs — base model zero-shot, then the LoRA-tuned model —
greedy decoding, identical prompts. Four metrics: tool-call
exact-match accuracy, argument accuracy, invalid-call rate, and
end-to-end task success, reported overall and per failure reason so the
silent-retry class is visible on its own.

## Results

First audit (Nov 8, 2026) — same harness, same 128-case held-out split,
full-precision base vs. LoRA-tuned, greedy decoding:

| Metric | Base (zero-shot) | Tuned v1 (LoRA) |
|---|---|---|
| Tool-call exact-match accuracy | 0.0288 (7/243) | 0.2099 (51/243) |
| Argument accuracy | 0.2523 (191/757) | 0.3725 (282/757) |
| Invalid-call rate | 0.0000 | 0.0000 |
| End-to-end task success (exact) | 0.0000 (0/128) | 0.0000 (0/128) |

The tuned model improved exact-match accuracy 7× and argument accuracy
by half — but end-to-end success stayed at zero, and the audit showed
exactly why. The first training run had a prompt-format mismatch: it
trained on raw user text while evaluation used the chat template, so at
eval time the model fell back to base-style pretty-printed JSON, never
emitted EOS after its plan, and kept appending extra calls (345
predicted vs. 243 gold). On the safety-critical silent class it was
worse than the base model: 0/17 bank-server-error cases stayed silent,
15/17 wrongly contacted the customer. The fix was a re-train with the
prompts rendered through the same chat template the evaluator uses, plus
an explicit EOS after each plan — one variable changed, same data,
same hyperparameters.

A second, subtler finding: the gold `message` text is selected by a
hash of the transaction ID, which the model cannot compute from the
prompt — so exact message match is chance-level (~1/3) and exact task
success is capped by it even for a perfect model. The audit therefore
also reports a structural task-success rate (same plan, ignoring the
message text): the reliability property this entry is actually about —
right tools, right order, right arguments, silence when required.

> *v2 re-evaluation (Nov 8, 2026, same split, same harness, tuned-only
> job):*

| Metric | Base (zero-shot) | Tuned v2 (LoRA, aligned prompts) |
|---|---|---|
| Tool-call exact-match accuracy | 0.0288 | **0.6667** (162/243) |
| Argument accuracy | 0.2523 | **0.8798** (666/757) |
| Invalid-call rate | 0.0000 | **0.0000** |
| End-to-end task success (exact) | 0.0000 (0/128) | **0.3672 (47/128)** |
| End-to-end task success (structural) | 0.0000 | **0.8438 (108/128)** |

The aligned-prompt retrain fixed the stopping problem outright: the
tuned model now emits exactly 243 calls for 243 gold calls (v1 emitted
345), zero parse failures, zero invalid calls, and the silent class is
17/17 — it never contacts the customer on a bank-server error. The
remaining errors are two plan-shape confusions (wrong-CVV cases
sometimes gain an extra offer call; insufficient-funds cases sometimes
lose the final offer call) and the hash-selected message texts, which
no model can predict from the prompt. Structural task success is 1.0
in 6 of 8 failure reasons.

Per-reason task success (tuned vs. base), and the silent-retry class
(bank_server_error: cases where the model stayed silent vs. wrongly
contacted the customer), are reported in the audit logs alongside the
headline table. Error analysis: the most common argument errors and
failure examples come straight from the scored predictions.

## Cost / ROI

The full fine-tune cost ~$0.21 of GPU credits (423 s on 1× L40S at
$1.80/hr); the before/after audit ran on the same flavor. A self-hosted
tuned 8B serves recovery decisions at the cost of one small GPU's
uptime, against per-token billing for a closed-source model on the same
volume — the tuned model is a fixed, auditable artifact a merchant can
run in-house, with every decision checkable against the rule engine
that taught it.

## Limits

Synthetic single-domain distribution (8 payment-failure reasons,
template-bound messages); one 8B base model; the audit measures
tool-call reliability on held-out cases from the same generator, and
transfer to live payment traffic is not claimed. The engine's per-rule
success-rate values are configured assumptions for the original demo's
simulation, never presented as measured results.

## Artifacts

- Adapter: [qasimjethwa/aya-expanse-8b-recovery-lora](https://huggingface.co/qasimjethwa/aya-expanse-8b-recovery-lora)
- Dataset + harness + eval runner: [qasimjethwa/recovery-copilot-toolcalls](https://huggingface.co/datasets/qasimjethwa/recovery-copilot-toolcalls)
- Demo: [Recovery Agent, Taught by Rules — HF Space](https://huggingface.co/spaces/qasimjethwa/recovery-agent-taught-by-rules) (recorded eval outputs for all 128 held-out cases; a Gradio Space with live inference was blocked because this account is not Pro — the static demo shows the same recorded outputs)
- Code: entry repository [qasimjethwa/recovery-agent-taught-by-rules](https://github.com/qasimjethwa/recovery-agent-taught-by-rules) (dataset generator, audit harness, training script, HF Jobs runners, results)
- Model collection: [Recovery Agent, Taught by Rules — Models](https://huggingface.co/collections/qasimjethwa/recovery-agent-taught-by-rules-models-6af684adaa2770453b61d2aa) · Dataset collection: [Recovery Agent, Taught by Rules — Datasets](https://huggingface.co/collections/qasimjethwa/recovery-agent-taught-by-rules-datasets-6af68542bda0286407d4e91b)
