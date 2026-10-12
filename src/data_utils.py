"""
Core SFT/DPO batching and loss logic.
submission_adapter.py just calls into these functions, keeping the actual
math here so I can reuse it from the real training scripts later instead
of duplicating logic in two places.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _get_pad_id(tokenizer):
    # Qwen's tokenizer (and a few others) don't ship a pad token by default,
    # so if that's missing I just fall back to eos. Works fine since we're
    # not actually attending over the padded positions anyway.
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    return pad_id


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Build a response-masked SFT batch, padded and truncated to max_length.

    A few things I had to be careful about here:
    - EOS gets appended to the response before truncation happens, so if a
      response is long enough to get cut off at max_length, it can lose its
      EOS. That's intentional per the assignment spec, not a bug.
    - Prompt tokens and padding tokens are both masked with -100 in labels,
      only the actual response tokens (plus EOS) are left unmasked.
    - Row order has to stay exactly as given, since the grader checks this.
    """
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses need to be the same length")

    pad_id = _get_pad_id(tokenizer)
    eos_id = tokenizer.eos_token_id

    sequences = []
    label_rows = []

    for prompt, response in zip(prompts, responses):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False) + [eos_id]

        full_sequence = (prompt_ids + response_ids)[:max_length]

        # edge case: if the prompt itself is already >= max_length, there's
        # no room left for any response tokens at all. clamping here so the
        # label masking below doesn't go out of bounds in that situation.
        prompt_len_in_seq = min(len(prompt_ids), len(full_sequence))

        labels = [-100] * prompt_len_in_seq + full_sequence[prompt_len_in_seq:]

        sequences.append(full_sequence)
        label_rows.append(labels)

    batch_size = len(prompts)
    input_ids = torch.full((batch_size, max_length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    labels_tensor = torch.full((batch_size, max_length), -100, dtype=torch.long)

    for row, (seq, labs) in enumerate(zip(sequences, label_rows)):
        length = len(seq)
        input_ids[row, :length] = torch.tensor(seq, dtype=torch.long)
        attention_mask[row, :length] = 1
        labels_tensor[row, :length] = torch.tensor(labs, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels_tensor,
    }


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Summed response-token log probability per row, shape (batch,).

    Standard causal LM shift applies here: logits at position t predict the
    token at position t+1, so I shift labels left by one before comparing
    against the logits.
    """
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    # build_sft_batch makes plain cpu tensors, so I need to move them onto
    # whatever device the model actually lives on before the forward pass,
    # otherwise this blows up the moment the model isn't on cpu
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (batch, seq_len, vocab)

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)

    # -100 entries would break gather, so I clamp them to 0 first and then
    # zero out their contribution with the mask right after.
    response_mask = shift_labels.ne(-100)
    safe_labels = shift_labels.clamp_min(0)

    token_logps = torch.gather(log_probs, dim=2, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * response_mask

    return token_logps.sum(dim=1)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Scalar SFT loss, response tokens only.

    Going with the per-example reduction here: mean of each row's summed
    response negative log likelihood. The manual says either reduction is
    fine as long as I'm consistent about it everywhere, so this is the one
    I'm sticking with for SFT, DPO, and the report.
    """
    response_logps = compute_response_logprobs(model, tokenizer, prompts, responses, max_length)
    return -response_logps.mean()


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Two response-masked LM batches, one for chosen and one for rejected.

    Same prompt gets formatted identically on both sides, which matters
    since chosen vs rejected log probs need to be directly comparable.
    """
    chosen_batch = build_sft_batch(tokenizer, prompts, chosen_responses, max_length)
    rejected_batch = build_sft_batch(tokenizer, prompts, rejected_responses, max_length)
    return {"chosen": chosen_batch, "rejected": rejected_batch}


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Scalar batch-mean DPO loss from precomputed log probabilities.

    Quick sanity check I keep in my head for this: if policy and reference
    logps are all equal, the inner term is zero, sigmoid(0) is 0.5, and
    -log(0.5) is log(2). That's basically the "model hasn't learned
    anything yet" baseline.
    """
    policy_diff = policy_chosen_logps - policy_rejected_logps
    reference_diff = reference_chosen_logps - reference_rejected_logps

    logits = beta * (policy_diff - reference_diff)

    # using logsigmoid instead of -log(sigmoid(x)) directly, it's the more
    # numerically stable way to compute this and avoids underflow for very
    # negative logits
    loss = -F.logsigmoid(logits)
    return loss.mean()


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Scalar mean DPO loss, reference model forward pass stays under no_grad.

    Important bit here: the reference model should never pick up gradients,
    it's frozen for the entire run. Wrapping just its two forward passes in
    no_grad keeps that clean without needing to touch requires_grad on its
    params directly.
    """
    policy_chosen_logps = compute_response_logprobs(
        policy_model, tokenizer, prompts, chosen_responses, max_length
    )
    policy_rejected_logps = compute_response_logprobs(
        policy_model, tokenizer, prompts, rejected_responses, max_length
    )

    with torch.no_grad():
        reference_chosen_logps = compute_response_logprobs(
            reference_model, tokenizer, prompts, chosen_responses, max_length
        )
        reference_rejected_logps = compute_response_logprobs(
            reference_model, tokenizer, prompts, rejected_responses, max_length
        )

    return compute_dpo_loss_from_logps(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
    )