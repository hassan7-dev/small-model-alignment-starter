"""
Pulls the actual datasets (UltraChat for SFT, HH-RLHF and UltraFeedback for
the two from-scratch DPO runs) and reshapes them into single-turn pairs.

Keeping this separate from data_utils.py since this file is about getting
raw data into the right shape, data_utils.py is about turning shaped data
into tensors. Different concerns.
"""

from __future__ import annotations

import re

from datasets import load_dataset

from src.prompting import format_prompt


def _first_exchange(messages):
    # ultrachat is mostly multi-turn conversations, if I require the WHOLE
    # conversation to be exactly one user turn and one assistant turn I end
    # up throwing out almost everything before getting anywhere close to
    # the dataset size I actually need. pulling out just the first user to
    # assistant exchange instead, which still gives me a clean single-turn
    # training pair without needing the rest of the conversation to match
    for i in range(len(messages) - 1):
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            return messages[i]["content"], messages[i + 1]["content"]
    return None


def load_sft_data(train_size=30000, val_size=1000, seed=42):
    """Load UltraChat, keep only single-turn exchanges, return (train, val)
    as lists of {"prompt": ..., "response": ...} dicts.
    """
    raw = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    raw = raw.shuffle(seed=seed)

    examples = []
    for row in raw:
        messages = row["messages"]
        exchange = _first_exchange(messages)
        if exchange is None:
            continue
        instruction, response = exchange
        instruction = instruction.strip()
        response = response.strip()
        if not instruction or not response:
            # skipping anything empty, cz no point training on a blank pair
            continue
        examples.append({"prompt": format_prompt(instruction), "response": response})
        if len(examples) >= train_size + val_size:
            break

    train = examples[:train_size]
    val = examples[train_size:train_size + val_size]
    return train, val


def _parse_hh_transcript(text):
    """HH-RLHF stores full conversations as one big string like:
    "\n\nHuman: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant: ..."
    I only want the single-turn ones, so this returns (human_turn,
    assistant_turn) if there's exactly one of each, otherwise None.
    """
    turns = re.split(r"\n\n(Human|Assistant): ", text)
    # turns[0] is usually an empty string before the first split, then it
    # alternates role, content, role, content...
    turns = [t for t in turns if t != ""]

    if len(turns) != 4:
        # more than one back-and-forth, not single turn, so we skip it
        return None

    role_a, content_a, role_b, content_b = turns
    if role_a != "Human" or role_b != "Assistant":
        return None

    return content_a.strip(), content_b.strip()


def load_hh_rlhf_dpo_data(train_size=15000, val_size=1000, seed=42):
    """Load the HH-RLHF preference pairs, filtered to single-turn exchanges.
    Returns (train, val) as lists of
    {"prompt": ..., "chosen": ..., "rejected": ...} dicts.
    """
    raw = load_dataset("Anthropic/hh-rlhf", split="train")
    raw = raw.shuffle(seed=seed)

    examples = []
    for row in raw:
        chosen_parsed = _parse_hh_transcript(row["chosen"])
        rejected_parsed = _parse_hh_transcript(row["rejected"])
        if chosen_parsed is None or rejected_parsed is None:
            continue

        chosen_prompt, chosen_response = chosen_parsed
        rejected_prompt, rejected_response = rejected_parsed

        # chosen and rejected should be answering the exact same question,
        # if the human turns don't match something's off with this row
        if chosen_prompt != rejected_prompt:
            continue
        if not chosen_response or not rejected_response:
            continue

        examples.append({
            "prompt": format_prompt(chosen_prompt),
            "chosen": chosen_response,
            "rejected": rejected_response,
        })
        if len(examples) >= train_size + val_size:
            break

    train = examples[:train_size]
    val = examples[train_size:train_size + val_size]
    return train, val


def _last_assistant_message(messages):
    # ultrafeedback_binarized stores chosen/rejected as message lists,
    # I just want the final assistant reply out of each
    for message in reversed(messages):
        if message["role"] == "assistant":
            return message["content"].strip()
    return None


def load_ultrafeedback_dpo_data(train_size=15000, val_size=1000, seed=42):
    """Load UltraFeedback preference pairs. Returns (train, val) as lists of
    {"prompt": ..., "chosen": ..., "rejected": ...} dicts.
    """
    raw = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    raw = raw.shuffle(seed=seed)

    examples = []
    for row in raw:
        instruction = row["prompt"].strip()
        chosen_response = _last_assistant_message(row["chosen"])
        rejected_response = _last_assistant_message(row["rejected"])

        if not instruction or not chosen_response or not rejected_response:
            continue
        if chosen_response == rejected_response:
            # happens occasionally in this dataset, not a useful pair if
            # both sides are identical
            continue

        examples.append({
            "prompt": format_prompt(instruction),
            "chosen": chosen_response,
            "rejected": rejected_response,
        })
        if len(examples) >= train_size + val_size:
            break

    train = examples[:train_size]
    val = examples[train_size:train_size + val_size]
    return train, val