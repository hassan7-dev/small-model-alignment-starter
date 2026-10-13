"""
Task 6. Generating responses from whatever checkpoint we point this at
(SFT, either DPO run, or the LoRA persona adapter), then building the
pairwise DPO-vs-SFT comparison files the judge will see in Task 7.

Reusing the manual checkpoint loaders from dpo.py here instead of
writing them twice, since the local-path hub validation bug that forced
us to write those in the first place applies just as much to generation
as it did to training. Same checkpoint, same disk layout, same problem.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file as load_safetensors
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dpo import _load_local_causal_lm, _load_local_tokenizer
from src.prompting import format_prompt

PERSONA_BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"


def build_alpaca_eval_subset(n=300, seed=42, output_path="eval_sets/alpaca_eval_300_seed42.json"):
    # pulling straight from the hub here, not a local directory, so none
    # of the path bug applies, this one's a normal from_pretrained style
    # download and just works
    dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]

    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    chosen_indices = sorted(indices[:n])

    rows = []
    for i in chosen_indices:
        row = dataset[i]
        rows.append({
            "id": f"alpaca-eval-42-{i:04d}",
            "instruction": row["instruction"],
            "dataset": "alpaca_eval",
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"wrote {len(rows)} alpaca eval prompts to {output_path}")
    return rows


def _load_qwen_checkpoint(checkpoint_path, device):
    # this covers the sft checkpoint and both from scratch dpo checkpoints,
    # they're all plain qwen2 checkpoints saved the same way
    tokenizer = _load_local_tokenizer(checkpoint_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_local_causal_lm(checkpoint_path, torch.float16)
    model.to(device)
    model.eval()
    return model, tokenizer


def _load_persona_lora_checkpoint(adapter_path, device):
    # base model comes straight from the hub by name, so that part's a
    # normal from_pretrained call with no local path involved. only the
    # adapter weights are local, so those get loaded by hand instead of
    # going through PeftModel.from_pretrained, same reasoning as the qwen
    # checkpoints above
    tokenizer = AutoTokenizer.from_pretrained(PERSONA_BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(PERSONA_BASE_MODEL_ID, torch_dtype=torch.float16)

    with open(os.path.join(adapter_path, "adapter_config.json")) as f:
        adapter_config_dict = json.load(f)
    # adapter_config.json has a few bookkeeping fields LoraConfig doesn't
    # accept as constructor args, so only pass through what it knows about
    lora_field_names = {field for field in LoraConfig.__dataclass_fields__}
    lora_kwargs = {k: v for k, v in adapter_config_dict.items() if k in lora_field_names}
    lora_config = LoraConfig(**lora_kwargs)

    model = get_peft_model(base_model, lora_config)
    state_dict = load_safetensors(os.path.join(adapter_path, "adapter_model.safetensors"))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected extra keys loading lora adapter: {unexpected}")

    model.to(device)
    model.eval()
    return model, tokenizer


def load_checkpoint(checkpoint_path, checkpoint_kind, device="cuda"):
    if checkpoint_kind == "qwen":
        return _load_qwen_checkpoint(checkpoint_path, device)
    elif checkpoint_kind == "persona_lora":
        return _load_persona_lora_checkpoint(checkpoint_path, device)
    else:
        raise ValueError(f"unknown checkpoint_kind: {checkpoint_kind}")


@torch.no_grad()
def generate_responses(
    model,
    tokenizer,
    eval_examples,
    generator_name,
    dataset_name,
    device="cuda",
    max_new_tokens=256,
    repetition_penalty=1.0,
    no_repeat_ngram_size=0,
    limit=None,
):
    # greedy decoding per the manual, task 2. repetition_penalty of 1.0
    # and no_repeat_ngram_size of 0 are both no-ops, only turn these on if
    # a model starts looping, and use the same settings for every model
    # being compared in one evaluation so the judge isn't just picking up
    # on different decoding behavior
    rows_to_run = eval_examples[:limit] if limit else eval_examples
    outputs = []

    for row in rows_to_run:
        prompt_text = format_prompt(row["instruction"])
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = generated_ids[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        outputs.append({
            "id": row["id"],
            "instruction": row["instruction"],
            "output": response_text,
            "generator": generator_name,
            "dataset": dataset_name,
        })

    return outputs


def run_generation(
    checkpoint_path,
    checkpoint_kind,
    eval_set_path,
    generator_name,
    dataset_name,
    output_path,
    device="cuda",
    max_new_tokens=256,
    repetition_penalty=1.0,
    no_repeat_ngram_size=0,
    sanity_check_n=None,
):
    with open(eval_set_path) as f:
        eval_examples = json.load(f)

    model, tokenizer = load_checkpoint(checkpoint_path, checkpoint_kind, device)

    outputs = generate_responses(
        model, tokenizer, eval_examples, generator_name, dataset_name,
        device=device, max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
        limit=sanity_check_n,
    )

    if sanity_check_n:
        # just print these out instead of saving, this is the manual's
        # "inspect a few responses by hand before running the full thing"
        # step, nothing to write to disk yet
        for row in outputs:
            print(f"\ninstruction: {row['instruction']}")
            print(f"output: {row['output']}")
        return outputs

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"wrote {len(outputs)} generations to {output_path}")
    return outputs


def build_pairwise_file(dpo_outputs_path, sft_outputs_path, output_path, seed=42):
    # randomizing which side each model lands on per pair, and keeping
    # the real mapping around in the same row so we can translate the
    # judge's A/B verdict back to dpo/sft afterward. the judge prompt in
    # task 7 should only ever include instruction, response_a, response_b,
    # never this mapping
    with open(dpo_outputs_path) as f:
        dpo_outputs = {row["id"]: row for row in json.load(f)}
    with open(sft_outputs_path) as f:
        sft_outputs = {row["id"]: row for row in json.load(f)}

    shared_ids = sorted(set(dpo_outputs) & set(sft_outputs))
    if len(shared_ids) != len(dpo_outputs) or len(shared_ids) != len(sft_outputs):
        print(f"warning: {len(dpo_outputs)} dpo rows and {len(sft_outputs)} sft rows, "
              f"only {len(shared_ids)} ids matched between them")

    rng = random.Random(seed)
    pairs = []
    for row_id in shared_ids:
        dpo_row = dpo_outputs[row_id]
        sft_row = sft_outputs[row_id]

        # coin flip per pair so the position doesn't always correlate
        # with which model produced it
        if rng.random() < 0.5:
            response_a, response_b = dpo_row["output"], sft_row["output"]
            position_to_model = {"A": "dpo", "B": "sft"}
        else:
            response_a, response_b = sft_row["output"], dpo_row["output"]
            position_to_model = {"A": "sft", "B": "dpo"}

        pairs.append({
            "id": row_id,
            "instruction": dpo_row["instruction"],
            "response_a": response_a,
            "response_b": response_b,
            "position_to_model": position_to_model,
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"wrote {len(pairs)} pairwise comparisons to {output_path}")
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    alpaca_parser = subparsers.add_parser("build-alpaca-subset")
    alpaca_parser.add_argument("--n", type=int, default=300)
    alpaca_parser.add_argument("--seed", type=int, default=42)
    alpaca_parser.add_argument("--output_path", type=str, default="eval_sets/alpaca_eval_300_seed42.json")

    gen_parser = subparsers.add_parser("generate")
    gen_parser.add_argument("--checkpoint_path", type=str, required=True)
    gen_parser.add_argument("--checkpoint_kind", type=str, choices=["qwen", "persona_lora"], required=True)
    gen_parser.add_argument("--eval_set_path", type=str, required=True)
    gen_parser.add_argument("--generator_name", type=str, required=True)
    gen_parser.add_argument("--dataset_name", type=str, required=True)
    gen_parser.add_argument("--output_path", type=str, required=True)
    gen_parser.add_argument("--max_new_tokens", type=int, default=256)
    gen_parser.add_argument("--repetition_penalty", type=float, default=1.0)
    gen_parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    gen_parser.add_argument("--sanity_check_n", type=int, default=None)

    pairs_parser = subparsers.add_parser("build-pairs")
    pairs_parser.add_argument("--dpo_outputs_path", type=str, required=True)
    pairs_parser.add_argument("--sft_outputs_path", type=str, required=True)
    pairs_parser.add_argument("--output_path", type=str, required=True)
    pairs_parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "build-alpaca-subset":
        build_alpaca_eval_subset(n=args.n, seed=args.seed, output_path=args.output_path)
    elif args.command == "generate":
        run_generation(
            checkpoint_path=args.checkpoint_path,
            checkpoint_kind=args.checkpoint_kind,
            eval_set_path=args.eval_set_path,
            generator_name=args.generator_name,
            dataset_name=args.dataset_name,
            output_path=args.output_path,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            sanity_check_n=args.sanity_check_n,
        )
    elif args.command == "build-pairs":
        build_pairwise_file(
            dpo_outputs_path=args.dpo_outputs_path,
            sft_outputs_path=args.sft_outputs_path,
            output_path=args.output_path,
            seed=args.seed,
        )