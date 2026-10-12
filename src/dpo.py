"""
From-scratch DPO training loop. Reused for both the HH-RLHF run and the
UltraFeedback run, just pass in a different data loading function and
output directory per call.

Following the manual's memory tip here: policy model on cuda:0, frozen
reference model on cuda:1, so a T4x2 setup doesn't run out of memory
holding two full copies of the model on one GPU.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data_utils import compute_response_logprobs, compute_dpo_loss_from_logps


def _reward_stats(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta):
    # these are the implicit rewards from the manual, section 10.5. not
    # the same thing as the loss, this is just for tracking how well the
    # policy is separating chosen from rejected
    rw = beta * (policy_chosen_logps - reference_chosen_logps)
    rl = beta * (policy_rejected_logps - reference_rejected_logps)
    accuracy = (rw > rl).float().mean().item()
    margin = (rw - rl).mean().item()
    return accuracy, margin


def _dpo_step(policy_model, reference_model, tokenizer, prompts, chosen, rejected, beta, max_length, policy_device, reference_device):
    # not reusing compute_dpo_loss from data_utils directly here since that
    # assumes policy and reference live on the same device, and I want them
    # split across cuda:0 and cuda:1. so I'm calling compute_response_logprobs
    # separately for each model and bringing the reference logps back onto
    # the policy's device before combining them
    policy_chosen_logps = compute_response_logprobs(policy_model, tokenizer, prompts, chosen, max_length)
    policy_rejected_logps = compute_response_logprobs(policy_model, tokenizer, prompts, rejected, max_length)

    with torch.no_grad():
        ref_chosen_logps = compute_response_logprobs(reference_model, tokenizer, prompts, chosen, max_length)
        ref_rejected_logps = compute_response_logprobs(reference_model, tokenizer, prompts, rejected, max_length)
        ref_chosen_logps = ref_chosen_logps.to(policy_device)
        ref_rejected_logps = ref_rejected_logps.to(policy_device)

    loss = compute_dpo_loss_from_logps(
        policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta
    )

    with torch.no_grad():
        accuracy, margin = _reward_stats(
            policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta
        )

    return loss, accuracy, margin


def evaluate_dpo(policy_model, reference_model, tokenizer, val_examples, beta, max_length, policy_device, reference_device, batch_size=2):
    policy_model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_margin = 0.0
    num_batches = 0

    with torch.no_grad():
        for i in range(0, len(val_examples), batch_size):
            chunk = val_examples[i:i + batch_size]
            prompts = [row["prompt"] for row in chunk]
            chosen = [row["chosen"] for row in chunk]
            rejected = [row["rejected"] for row in chunk]

            # same fix as in the sft eval, this one's actually more memory
            # hungry since it's doing four forward passes per batch
            # (policy and reference, chosen and rejected)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss, accuracy, margin = _dpo_step(
                    policy_model, reference_model, tokenizer, prompts, chosen, rejected,
                    beta, max_length, policy_device, reference_device,
                )
            total_loss += loss.item()
            total_accuracy += accuracy
            total_margin += margin
            num_batches += 1
            del loss
            torch.cuda.empty_cache()

    policy_model.train()
    num_batches = max(1, num_batches)
    return {
        "loss": total_loss / num_batches,
        "reward_accuracy": total_accuracy / num_batches,
        "reward_margin": total_margin / num_batches,
    }


def train_dpo(
    sft_checkpoint_path,
    output_dir,
    load_preference_data_fn,
    train_size=15000,
    val_size=1000,
    context_length=512,
    beta=0.1,
    learning_rate=1e-6,
    per_device_batch_size=1,
    grad_accum_steps=32,
    grad_clip_norm=1.0,
    num_epochs=1,
    seed=42,
    eval_every_steps=100,
    policy_device="cuda:0",
    reference_device="cuda:1",
):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)

    print(f"loading policy from {sft_checkpoint_path} onto {policy_device}")
    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # policy stays fp32 since I'm doing grad scaling on it, reference can be
    # fp16 since it never gets a gradient anyway, just saves memory
    policy_model = AutoModelForCausalLM.from_pretrained(sft_checkpoint_path, torch_dtype=torch.float32)
    policy_model.to(policy_device)
    policy_model.train()

    print(f"loading frozen reference copy onto {reference_device}")
    reference_model = AutoModelForCausalLM.from_pretrained(sft_checkpoint_path, torch_dtype=torch.float16)
    reference_model.to(reference_device)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    print("loading preference data...")
    train_examples, val_examples = load_preference_data_fn(train_size=train_size, val_size=val_size, seed=seed)
    print(f"got {len(train_examples)} train pairs, {len(val_examples)} val pairs")

    optimizer = AdamW(policy_model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    best_reward_accuracy = -1.0
    optim_step = 0
    micro_step = 0

    print("starting DPO training")
    start_time = time.time()

    for epoch in range(num_epochs):
        random.shuffle(train_examples)
        optimizer.zero_grad()

        for i in range(0, len(train_examples), per_device_batch_size):
            chunk = train_examples[i:i + per_device_batch_size]
            if not chunk:
                continue
            prompts = [row["prompt"] for row in chunk]
            chosen = [row["chosen"] for row in chunk]
            rejected = [row["rejected"] for row in chunk]

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss, _, _ = _dpo_step(
                    policy_model, reference_model, tokenizer, prompts, chosen, rejected,
                    beta, context_length, policy_device, reference_device,
                )
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            micro_step += 1

            if micro_step % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                optim_step += 1

                if optim_step % eval_every_steps == 0:
                    val_stats = evaluate_dpo(
                        policy_model, reference_model, tokenizer, val_examples,
                        beta, context_length, policy_device, reference_device,
                    )
                    elapsed = time.time() - start_time
                    record = {
                        "step": optim_step,
                        "train_loss": loss.item() * grad_accum_steps,
                        "val_loss": val_stats["loss"],
                        "val_reward_accuracy": val_stats["reward_accuracy"],
                        "val_reward_margin": val_stats["reward_margin"],
                        "elapsed_seconds": elapsed,
                    }
                    print(
                        f"step {optim_step}: val_loss={val_stats['loss']:.4f} "
                        f"reward_acc={val_stats['reward_accuracy']:.3f} "
                        f"reward_margin={val_stats['reward_margin']:.4f}"
                    )

                    with open(metrics_path, "a") as f:
                        f.write(json.dumps(record) + "\n")

                    # checkpoint selection is on reward accuracy, not loss,
                    # that's what the manual asks for in section 10.1
                    if val_stats["reward_accuracy"] > best_reward_accuracy:
                        best_reward_accuracy = val_stats["reward_accuracy"]
                        best_path = os.path.join(output_dir, "best")
                        policy_model.save_pretrained(best_path)
                        tokenizer.save_pretrained(best_path)
                        print(f"new best reward_accuracy {best_reward_accuracy:.3f}, saved to {best_path}")

    final_path = os.path.join(output_dir, "final")
    policy_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"DPO training done, final checkpoint saved to {final_path}")

    return {"best_reward_accuracy": best_reward_accuracy, "output_dir": output_dir}


if __name__ == "__main__":
    from src.data_loading import load_hh_rlhf_dpo_data, load_ultrafeedback_dpo_data

    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=str, choices=["hh_rlhf", "ultrafeedback"], required=True)
    parser.add_argument("--sft_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    data_fn = load_hh_rlhf_dpo_data if args.track == "hh_rlhf" else load_ultrafeedback_dpo_data

    train_dpo(
        sft_checkpoint_path=args.sft_checkpoint,
        output_dir=args.output_dir,
        load_preference_data_fn=data_fn,
    )