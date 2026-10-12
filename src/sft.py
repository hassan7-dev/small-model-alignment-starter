"""
From-scratch SFT training loop for Qwen2.5-0.5B. No HF Trainer here, per
the assignment's library boundary, just a plain PyTorch loop built on top
of the batching/loss functions in data_utils.py.

Meant to be run on Kaggle (T4 GPUs), but writing it as a normal importable
module with a main guard so it's not tied to notebook cell state.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data_loading import load_sft_data
from src.data_utils import compute_sft_loss


class SFTDataset(Dataset):
    # just a thin wrapper, each item is a dict with "prompt" and "response"
    # so the collate function below can build a proper batch out of it
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def _collate(batch):
    prompts = [row["prompt"] for row in batch]
    responses = [row["response"] for row in batch]
    return prompts, responses


def _cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    # writing this by hand instead of pulling in get_cosine_schedule, it's
    # like five lines and I'd rather not add another import for it
    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def evaluate(model, tokenizer, val_examples, max_length, device, batch_size=4):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for i in range(0, len(val_examples), batch_size):
            chunk = val_examples[i:i + batch_size]
            prompts = [row["prompt"] for row in chunk]
            responses = [row["response"] for row in chunk]
            # matching the same autocast precision training uses, otherwise
            # eval runs the full fp32 logits tensor through log_softmax and
            # that's a big memory spike with qwen's vocab size, enough to
            # OOM a T4 on its own
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = compute_sft_loss(model, tokenizer, prompts, responses, max_length)
            total_loss += loss.item()
            num_batches += 1
            # clearing this out after every eval batch so memory doesn't
            # slowly build up across a run with a lot of eval checkpoints
            del loss
            torch.cuda.empty_cache()
    model.train()
    return total_loss / max(1, num_batches)


def train_sft(
    output_dir,
    model_name="Qwen/Qwen2.5-0.5B",
    train_size=30000,
    val_size=1000,
    context_length=512,
    per_device_batch_size=2,
    grad_accum_steps=16,
    learning_rate=2e-5,
    warmup_ratio=0.03,
    weight_decay=0.1,
    grad_clip_norm=1.0,
    num_epochs=1,
    seed=42,
    eval_every_steps=200,
    device="cuda",
):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)

    print("loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        # Qwen doesn't ship a pad token out of the box, falling back to eos
        tokenizer.pad_token = tokenizer.eos_token

    # keeping the trainable weights in fp32 since I'm using grad scaling for
    # mixed precision, autocast handles the fp16 compute during forward/backward
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.to(device)
    model.train()

    print("loading and formatting SFT data...")
    train_examples, val_examples = load_sft_data(train_size=train_size, val_size=val_size, seed=seed)
    print(f"got {len(train_examples)} train examples, {len(val_examples)} val examples")

    train_dataset = SFTDataset(train_examples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        shuffle=True,
        collate_fn=_collate,
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_optim_steps = steps_per_epoch * num_epochs
    warmup_steps = max(1, int(total_optim_steps * warmup_ratio))
    scheduler = _cosine_schedule_with_warmup(optimizer, warmup_steps, total_optim_steps)

    scaler = torch.cuda.amp.GradScaler()

    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    best_val_loss = float("inf")
    optim_step = 0
    micro_step = 0

    print(f"starting training, {total_optim_steps} optimizer steps planned")
    start_time = time.time()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        for prompts, responses in train_loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = compute_sft_loss(model, tokenizer, prompts, responses, context_length)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            micro_step += 1

            if micro_step % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                optim_step += 1

                if optim_step % eval_every_steps == 0:
                    val_loss = evaluate(model, tokenizer, val_examples, context_length, device)
                    elapsed = time.time() - start_time
                    record = {
                        "step": optim_step,
                        "train_loss": loss.item() * grad_accum_steps,
                        "val_loss": val_loss,
                        "lr": scheduler.get_last_lr()[0],
                        "elapsed_seconds": elapsed,
                    }
                    print(f"step {optim_step}: train_loss={record['train_loss']:.4f} val_loss={val_loss:.4f}")

                    # writing after every eval so a kaggle disconnect doesn't
                    # wipe out progress I already have
                    with open(metrics_path, "a") as f:
                        f.write(json.dumps(record) + "\n")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_path = os.path.join(output_dir, "best")
                        model.save_pretrained(best_path)
                        tokenizer.save_pretrained(best_path)
                        print(f"new best val_loss {val_loss:.4f}, saved to {best_path}")

    final_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"training done, final checkpoint saved to {final_path}")

    return {"best_val_loss": best_val_loss, "output_dir": output_dir}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/sft_run")
    parser.add_argument("--train_size", type=int, default=30000)
    parser.add_argument("--val_size", type=int, default=1000)
    args = parser.parse_args()

    train_sft(
        output_dir=args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
    )