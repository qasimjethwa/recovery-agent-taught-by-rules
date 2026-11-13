# Submission record — Lexsi AI Safety & Alignment Hackathon

**Submitted:** Fri 13 Nov 2026, 12:05 PM IST (06:35 UTC)
**Entrant:** Qasim Jethwa (solo), registered email qasimjethwa9783@gmail.com,
registration ticket #W6MH-XAC1-Y5XV
**Track:** 3 — Agentic Fine-tuning
**Project:** Recovery Agent, Taught by Rules

## Confirmation

- On-screen: "Submission received!" — "Thank you. A confirmation with
  your abstract and Discord username is on its way to your email."
  "To fix anything, submit the form again. Your latest submission counts."
- Email receipt: "Lexsi Hackathon: submission received", from Lexsi Labs
  <notifications@tally.so>, 13 Nov 2026 12:05:12 IST, to
  qasimjethwa9783@gmail.com; reply-to ram.kadiyala@lexsi.ai. The email
  repeats the submitted abstract verbatim. No reference number or
  submission ID was issued (on screen or by email).
- Form: https://hackathon.lexsi.ai/submission/ (4 steps). Deadline:
  Nov 15, 2026, 11:59 PM AoE. Judging Nov 15–16; shortlist (up to 20)
  announced by Nov 16; thesis-proof video/slides due Nov 22 if shortlisted.

## What was filed

- Email: qasimjethwa9783@gmail.com; abstract pasted exactly as drafted
  (~370 words); Discord username left blank.
- HF model collection: https://huggingface.co/collections/qasimjethwa/recovery-agent-taught-by-rules-models-6af684adaa2770453b61d2aa
- HF dataset collection: https://huggingface.co/collections/qasimjethwa/recovery-agent-taught-by-rules-datasets-6af68542bda0286407d4e91b
- GitHub repo: https://github.com/qasimjethwa/recovery-agent-taught-by-rules
- Write-up: the GitHub repo README (full article text in `article.md`).
  No HF Article exists — HF article publishing requires a Pro
  subscription this account does not have; the hackathon rules accept a
  GitHub repo README as the write-up, and the form accepted the README
  URL in the article field. No article URL was invented.
- Repos/toolkits: AgentTune + AuditKIT. Model: Aya Expanse 8B.
- Problem statements (form allows at most 2): 1. Reliable tool calling,
  2. Efficient trajectories.
- Consent: only the "All repos and collections are public and final as of
  Nov 15, 11:59 PM AoE" attestation was ticked; the two promotional
  consents (sharing/promotion by Hugging Face and Cohere; being tagged
  for winning/featured submissions) were left unticked.
- The form has no fields for project title, ticket ID, team size, demo
  Space URL, adapter URL or dataset URL — those are covered via the
  collection links and the abstract only.

## Public links (all verified live 13 Nov 2026)

- Repo: https://github.com/qasimjethwa/recovery-agent-taught-by-rules
- Adapter: https://huggingface.co/qasimjethwa/aya-expanse-8b-recovery-lora
- Dataset: https://huggingface.co/datasets/qasimjethwa/recovery-copilot-toolcalls
- Demo: https://huggingface.co/spaces/qasimjethwa/recovery-agent-taught-by-rules
- Model collection: see above; dataset collection: see above.

## Results filed (real, measured Nov 8, same 128-case held-out split)

| Metric | Base (zero-shot) | Tuned v2 (LoRA) |
|---|---|---|
| Tool-call exact-match accuracy | 0.0288 | 0.6667 (162/243) |
| Argument accuracy | 0.2523 | 0.8798 (666/757) |
| Invalid-call rate | 0.0000 | 0.0000 |
| End-to-end task success (exact) | 0.0000 (0/128) | 0.3672 (47/128) |
| End-to-end task success (structural) | 0.0000 | 0.8438 (108/128) |

Silent class (bank_server_error): 17/17. Total cost ~$2.13 of the $20 HF credits.
