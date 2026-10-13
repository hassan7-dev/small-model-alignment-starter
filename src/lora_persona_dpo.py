"""
Task 5, the idiot sandwich one. Training a LoRA adapter on top
of SmolLM2-360M-Instruct so it talks like Gordon Ramsay, using the
provided persona preference pairs.

I kept this separate from dpo.py on purpose since the two tasks have
basically nothing in common except both being "DPO" in name. One's a
from scratch implementation on the big preference sets, this one is a
quick TRL run on a tiny 220 row dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"


def load_persona_pairs(data_path, val_fraction=0.1, seed=42):
    # the file is just a flat list of {prompt, chosen, rejected} dicts,
    # 220 rows total per the manual, nothing fancy to parse here
    with open(data_path) as f:
        rows = json.load(f)

    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)

    val_size = max(1, int(len(shuffled) * val_fraction))
    val_rows = shuffled[:val_size]
    train_rows = shuffled[val_size:]

    print(f"loaded {len(rows)} persona pairs, split into {len(train_rows)} train / {len(val_rows)} val")
    return train_rows, val_rows


def _to_conversational_dataset(rows):
    # TRL's DPOTrainer knows how to handle conversational format out of
    # the box as long as the tokenizer has a chat template, which
    # SmolLM2-Instruct does. so instead of hand formatting strings like
    # in src/prompting.py, just hand it message lists and let TRL apply
    # the template itself. this keeps the persona run consistent with
    # however SmolLM2 was actually instruction tuned
    prompts = []
    chosens = []
    rejecteds = []
    for row in rows:
        prompts.append([{"role": "user", "content": row["prompt"]}])
        chosens.append([{"role": "assistant", "content": row["chosen"]}])
        rejecteds.append([{"role": "assistant", "content": row["rejected"]}])

    return Dataset.from_dict({"prompt": prompts, "chosen": chosens, "rejected": rejecteds})


def train_lora_persona_dpo(
    data_path="data/persona_lora/ramsay_style_preferences.json",
    output_dir="outputs/lora_persona_run",
    beta=0.1,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    max_prompt_length=256,
    max_length=512,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    seed=42,
):
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"loading base model {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)

    train_rows, val_rows = load_persona_pairs(data_path, seed=seed)
    train_dataset = _to_conversational_dataset(train_rows)
    val_dataset = _to_conversational_dataset(val_rows)

    # SmolLM2 is a Llama style architecture under the hood, so the usual
    # attention projection names apply here. going with a small rank
    # since this is just steering style on a 360M model, not teaching it
    # new knowledge, a bigger rank would just risk overfitting 220 pairs
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )

    training_args = DPOConfig(
        output_dir=output_dir,
        beta=beta,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        # newer trl folded max_prompt_length into just max_length, with
        # truncation_mode controlling which side gets cut if a sequence
        # runs over. keep_start keeps the prompt intact and trims from
        # the end if needed, which is what we want here
        max_length=max_length,
        truncation_mode="keep_start",
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="no",
        bf16=torch.cuda.is_available(),
        report_to=[],
        seed=seed,
    )

    # not passing a ref_model here on purpose. when you give DPOTrainer a
    # peft_config and no explicit ref_model, it uses the frozen base
    # weights (adapters disabled) as the reference automatically, so we
    # don't need to load a second full copy of the model like we did in
    # dpo.py for the from scratch runs
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    final_path = os.path.join(output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"lora persona DPO done, adapter saved to {final_path}")

    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    with open(metrics_path, "w") as f:
        for entry in trainer.state.log_history:
            f.write(json.dumps(entry) + "\n")

    return {"output_dir": output_dir, "final_checkpoint": final_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/persona_lora/ramsay_style_preferences.json")
    parser.add_argument("--output_dir", type=str, default="outputs/lora_persona_run")
    args = parser.parse_args()

    train_lora_persona_dpo(data_path=args.data_path, output_dir=args.output_dir)