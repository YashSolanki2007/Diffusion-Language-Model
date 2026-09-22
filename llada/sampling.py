from __future__ import annotations

import math

import torch

from .model import LLaDA


def _predict_tokens(
    logits: torch.Tensor,
    mask_id: int,
    temperature: float,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return token predictions and their model confidence at every position."""
    logits = logits.float().clone()
    # [MASK] is a corruption token, never a clean-data target.
    logits[..., mask_id] = -torch.inf
    if top_k > 0:
        k = min(top_k, logits.size(-1) - 1)
        threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits.masked_fill_(logits < threshold, -torch.inf)

    if temperature <= 0:
        probabilities = torch.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
    else:
        probabilities = torch.softmax(logits / temperature, dim=-1)
        predictions = torch.multinomial(
            probabilities.view(-1, probabilities.size(-1)), 1
        ).view(probabilities.shape[:-1])
    confidence = probabilities.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)
    return predictions, confidence


@torch.inference_mode()
def generate(
    model: LLaDA,
    *,
    mask_id: int,
    prompt_tokens: torch.Tensor,
    generation_length: int,
    steps: int,
    remasking: str = "low_confidence",
    temperature: float = 0.0,
    top_k: int = 0,
    show_progress: bool = False,
) -> torch.Tensor:
    """Generate with Algorithm 4 or 5 from the LLaDA paper.

    The optional prompt is clamped throughout the reverse process. Returned tokens include it.
    """
    if prompt_tokens.ndim != 2:
        raise ValueError("prompt_tokens must have shape [batch, prompt_length]")
    if generation_length < 1:
        raise ValueError("generation_length must be at least 1")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if remasking not in {"low_confidence", "random"}:
        raise ValueError("remasking must be 'low_confidence' or 'random'")
    total_length = prompt_tokens.size(1) + generation_length
    if total_length > model.config.max_seq_len:
        raise ValueError(
            f"prompt + generation length ({total_length}) exceeds the checkpoint's "
            f"maximum sequence length ({model.config.max_seq_len})"
        )

    batch_size, prompt_length = prompt_tokens.shape
    generated = torch.full(
        (batch_size, generation_length),
        mask_id,
        dtype=torch.long,
        device=prompt_tokens.device,
    )
    tokens = torch.cat((prompt_tokens, generated), dim=1)
    generated_slice = slice(prompt_length, total_length)

    was_training = model.training
    model.eval()
    try:
        for step in range(steps):
            t = 1.0 - step / steps
            s = 1.0 - (step + 1) / steps
            currently_masked = tokens.eq(mask_id)
            logits = model(tokens)
            predictions, confidence = _predict_tokens(logits, mask_id, temperature, top_k)
            proposals = torch.where(currently_masked, predictions, tokens)

            if remasking == "random":
                remask_probability = s / t
                remask = (torch.rand_like(confidence) < remask_probability) & currently_masked
                next_tokens = torch.where(remask, mask_id, proposals)
                next_tokens[:, :prompt_length] = prompt_tokens
                tokens = next_tokens
            else:
                # Keep exactly floor(L * (1-s)) generated positions. Previously accepted
                # positions receive infinite confidence, so the reverse trajectory is monotonic.
                generated_confidence = confidence[:, generated_slice].clone()
                already_unmasked = ~currently_masked[:, generated_slice]
                generated_confidence.masked_fill_(already_unmasked, torch.inf)
                keep_count = min(generation_length, math.floor(generation_length * (1.0 - s)))
                if step == steps - 1:
                    keep_count = generation_length
                keep_indices = generated_confidence.topk(keep_count, dim=1).indices
                keep = torch.zeros_like(generated_confidence, dtype=torch.bool)
                keep.scatter_(1, keep_indices, True)
                tokens[:, generated_slice] = torch.where(
                    keep, proposals[:, generated_slice], mask_id
                )

            if show_progress:
                remaining = tokens[:, generated_slice].eq(mask_id).sum(dim=1).tolist()
                print(f"step {step + 1:4d}/{steps}: masks remaining {remaining}")
    finally:
        model.train(was_training)

    if tokens[:, generated_slice].eq(mask_id).any():
        raise RuntimeError("reverse process ended with unresolved mask tokens")
    return tokens

