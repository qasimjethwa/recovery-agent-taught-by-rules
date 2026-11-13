#!/usr/bin/env python3
"""
LoRA SFT for the Lexsi Track 3 (Agentic Fine-tuning) entry — AgentTune step.

Fine-tunes Aya Expanse 8B (Cohere Labs) on the Recovery Copilot tool-calling
dataset so the model learns to execute payment-recovery plans with the right
tools, in the right order, with the right arguments — and to stay silent when
the correct action is a silent backend retry.

STATUS: prepared pre-window; RUN IN-WINDOW (Nov 1–14, 2026). The run itself,
the base-model baseline and all reported metrics happen inside the build
window. Nothing in this file fabricates results.

Setup (in-window):
    pip install torch transformers datasets peft trl accelerate pyyaml
    export HF_TOKEN=<your Hugging Face token>     # placeholder — fill in-window
    python3 train_lora.py --config config.yaml

Data format: each dataset row becomes one chat example — the scenario as the
user turn (with the tool schemas), the gold tool-call sequence as the
assistant turn (JSON list). Loss is applied to the assistant turn only
(completion-only via TRL's SFTTrainer prompt/completion format).
"""

import argparse
import json
import os
import sys

SYSTEM_PROMPT = (
    "You are a payment-recovery agent. Given a failed payment and its diagnosed "
    "failure reason, respond ONLY with a JSON list of tool calls that executes "
    "the correct recovery plan, in order. Each call is an object with 'tool' "
    "and 'arguments'. Do not add commentary."
)


def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def format_example(row):
    user = (
        row["instruction"]
        + "\n\nAvailable tools:\n"
        + json.dumps(row["tools"], ensure_ascii=False)
    )
    completion = json.dumps(row["expected_calls"], ensure_ascii=False)
    return {"prompt": user, "completion": completion}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN is not set. In-window step: export your Hugging Face "
                 "token first (see config.yaml). Refusing to run unauthenticated.")

    # Imports are deferred so this file stays importable (and reviewable)
    # on machines without the training stack installed.
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    here = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(here, cfg["train_file"])
    rows = [json.loads(l) for l in open(train_path)]
    dataset = Dataset.from_list([format_example(r) for r in rows])
    print(f"training examples: {len(dataset)}")

    model_id = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto")

    lora = cfg["lora"]
    peft_config = LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], task_type=lora["task_type"])

    t = cfg["training"]
    sft_config = SFTConfig(
        output_dir=os.path.join(here, cfg["output"]["local_dir"]),
        num_train_epochs=t["num_epochs"],
        per_device_train_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler"],
        warmup_ratio=t["warmup_ratio"],
        max_length=t["max_seq_length"],
        logging_steps=t["logging_steps"],
        save_strategy=t["save_strategy"],
        bf16=t["bf16"],
        completion_only_loss=True,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model, args=sft_config, train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(sft_config.output_dir)
    tokenizer.save_pretrained(sft_config.output_dir)
    print(f"adapter saved to {sft_config.output_dir}")

    adapter_repo = cfg["output"].get("adapter_repo")
    if adapter_repo:
        trainer.push_to_hub(adapter_repo)
        print(f"adapter pushed to https://huggingface.co/{adapter_repo}")
    else:
        print("output.adapter_repo is null in config — skipping Hub upload "
              "(set it in-window to publish the adapter artifact).")


if __name__ == "__main__":
    main()
