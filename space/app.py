"""Recovery Agent, Taught by Rules — HF Space demo (Lexsi Track 3).

Shows the entry's REAL evaluation outputs: for each held-out case, the
gold recovery plan, the base model's prediction and the LoRA-tuned
model's prediction, with the audit verdict for each. No live inference
here — every number shown is a recorded output of the Nov 8 evaluation
run (predictions_*.jsonl + results_*.json in this repo), so the demo
cannot hallucinate a result. Live inference (base + adapter on GPU) is
the upgrade path; see app_live.py.
"""

import json
import os

import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


EVAL = {r["id"]: r for r in load_jsonl(os.path.join(HERE, "eval.jsonl"))}
BASE = {r["id"]: r for r in load_jsonl(os.path.join(HERE, "predictions_baseline.jsonl"))}
TUNED = {r["id"]: r for r in load_jsonl(os.path.join(HERE, "predictions_tuned.jsonl"))}
with open(os.path.join(HERE, "results_baseline.json")) as f:
    RES_BASE = json.load(f)
with open(os.path.join(HERE, "results_tuned.json")) as f:
    RES_TUNED = json.load(f)

REASON_LABELS = sorted({r["scenario"]["failure_reason"] for r in EVAL.values()})


def verdict(gold, pred):
    return "✅ exact match" if pred == gold else "❌ differs from gold"


def show_case(case_id):
    row = EVAL[case_id]
    gold = row["expected_calls"]
    b = BASE[case_id]["predicted_calls"]
    t = TUNED[case_id]["predicted_calls"]
    scenario = (
        f"Transaction {row['scenario']['txn_id']}: {row['scenario']['customer']}, "
        f"{row['scenario']['payment_method']}, ₹{row['scenario']['amount_inr']}. "
        f"Diagnosed failure: {row['scenario']['failure_label']}."
    )
    return (
        scenario,
        json.dumps(gold, indent=2),
        json.dumps(b, indent=2),
        verdict(gold, b),
        json.dumps(t, indent=2),
        verdict(gold, t),
    )


def metrics_table():
    def fmt(res):
        m = res["metrics"]
        return (f"exact-match {m['call_exact_match_accuracy']:.4f} · "
                f"arguments {m['argument_accuracy']:.4f} · "
                f"invalid {m['invalid_call_rate']:.4f} · "
                f"task success {m['task_success_rate']:.4f} "
                f"({res['counts']['successful_cases']}/{res['cases']})")
    return (f"**Base (Aya Expanse 8B, zero-shot):** {fmt(RES_BASE)}\n\n"
            f"**Tuned (LoRA adapter):** {fmt(RES_TUNED)}")


CASE_CHOICES = [
    (f"{cid} — {EVAL[cid]['scenario']['failure_label']}", cid)
    for cid in sorted(EVAL)
]

with gr.Blocks(title="Recovery Agent, Taught by Rules") as demo:
    gr.Markdown(
        "# Recovery Agent, Taught by Rules\n"
        "Fine-tuned Aya Expanse 8B for reliable tool use in failed-payment "
        "recovery — Lexsi AI Safety & Alignment Hackathon, Track 3.\n\n"
        "Pick a held-out eval case and compare: the **gold** recovery plan "
        "(from the deterministic Recovery Copilot rule engine), the **base** "
        "model's zero-shot plan, and the **tuned** model's plan. "
        "Every prediction shown is a recorded output of the Nov 8 "
        "evaluation run — nothing here is generated live or invented."
    )
    gr.Markdown(metrics_table())
    case = gr.Dropdown(choices=CASE_CHOICES, value=CASE_CHOICES[0][1],
                       label="Held-out eval case")
    scenario = gr.Textbox(label="Scenario", interactive=False)
    with gr.Row():
        gold = gr.Code(label="Gold plan (rule engine)", language="json")
        base = gr.Code(label="Base model prediction", language="json")
        tuned = gr.Code(label="Tuned model prediction", language="json")
    with gr.Row():
        base_v = gr.Textbox(label="Base verdict", interactive=False)
        tuned_v = gr.Textbox(label="Tuned verdict", interactive=False)
    case.change(show_case, inputs=case,
                outputs=[scenario, gold, base, base_v, tuned, tuned_v])
    demo.load(show_case, inputs=case,
              outputs=[scenario, gold, base, base_v, tuned, tuned_v])

if __name__ == "__main__":
    demo.launch()
