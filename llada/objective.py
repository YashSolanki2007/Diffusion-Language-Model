from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DiffusionBatch:
    clean_tokens: torch.Tensor
    noisy_tokens: torch.Tensor
    masked_positions: torch.Tensor
    timesteps: torch.Tensor


@dataclass
class SFTDiffusionBatch:
    clean_tokens: torch.Tensor
    noisy_tokens: torch.Tensor
    response_positions: torch.Tensor
    masked_positions: torch.Tensor
    timesteps: torch.Tensor


def make_diffusion_batch(
    clean_tokens: torch.Tensor,
    mask_id: int,
    *,
    generator: torch.Generator | None = None,
    min_t: float = 1e-5,
) -> DiffusionBatch:
    """Sample q(x_t | x_0): independently mask each token with probability t."""
    if clean_tokens.ndim != 2:
        raise ValueError("clean_tokens must have shape [batch, sequence]")
    batch_size = clean_tokens.size(0)
    # U(0, 1] in the paper. Clamping only excludes an unstable machine-zero endpoint.
    timesteps = torch.rand(
        (batch_size, 1), device=clean_tokens.device, generator=generator
    ).clamp_min(min_t)
    masked_positions = torch.rand(
        clean_tokens.shape, device=clean_tokens.device, generator=generator
    ) < timesteps
    noisy_tokens = clean_tokens.masked_fill(masked_positions, mask_id)
    return DiffusionBatch(clean_tokens, noisy_tokens, masked_positions, timesteps)


def llada_pretraining_loss(
    logits: torch.Tensor,
    batch: DiffusionBatch,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Monte Carlo estimator of Eq. (3): CE on masks, normalized by t * L."""
    if logits.shape[:2] != batch.clean_tokens.shape:
        raise ValueError("logits and token batch shapes do not match")
    token_loss = F.cross_entropy(
        logits.transpose(1, 2), batch.clean_tokens, reduction="none"
    )
    masked_float = batch.masked_positions.to(token_loss.dtype)
    seq_len = batch.clean_tokens.size(1)
    per_example = (token_loss * masked_float).sum(dim=1) / (
        batch.timesteps.squeeze(1) * seq_len
    )
    loss = per_example.mean()

    with torch.no_grad():
        predictions = logits.argmax(dim=-1)
        masked_count = batch.masked_positions.sum()
        correct = ((predictions == batch.clean_tokens) & batch.masked_positions).sum()
        accuracy = correct.float() / masked_count.clamp_min(1)
        metrics = {
            "loss": loss.detach(),
            "masked_accuracy": accuracy,
            "mask_fraction": batch.masked_positions.float().mean(),
            "mean_t": batch.timesteps.mean(),
        }
    return loss, metrics


def make_sft_diffusion_batch(
    clean_tokens: torch.Tensor,
    response_positions: torch.Tensor,
    mask_id: int,
    *,
    generator: torch.Generator | None = None,
    min_t: float = 1e-5,
) -> SFTDiffusionBatch:
    """Mask response tokens only, leaving the entire prompt visible."""
    if clean_tokens.shape != response_positions.shape:
        raise ValueError("clean_tokens and response_positions must have the same shape")
    batch_size = clean_tokens.size(0)
    timesteps = torch.rand(
        (batch_size, 1), device=clean_tokens.device, generator=generator
    ).clamp_min(min_t)
    masked_positions = (
        torch.rand(clean_tokens.shape, device=clean_tokens.device, generator=generator) < timesteps
    ) & response_positions
    noisy_tokens = clean_tokens.masked_fill(masked_positions, mask_id)
    return SFTDiffusionBatch(
        clean_tokens, noisy_tokens, response_positions, masked_positions, timesteps
    )


def llada_sft_loss(
    logits: torch.Tensor,
    batch: SFTDiffusionBatch,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Monte Carlo estimator of the response-only SFT objective in Equation (5)."""
    token_loss = F.cross_entropy(
        logits.transpose(1, 2), batch.clean_tokens, reduction="none"
    )
    masked_float = batch.masked_positions.to(token_loss.dtype)
    response_lengths = batch.response_positions.sum(dim=1).clamp_min(1)
    per_example = (token_loss * masked_float).sum(dim=1) / (
        batch.timesteps.squeeze(1) * response_lengths
    )
    loss = per_example.mean()
    with torch.no_grad():
        predictions = logits.argmax(dim=-1)
        masked_count = batch.masked_positions.sum()
        correct = ((predictions == batch.clean_tokens) & batch.masked_positions).sum()
        metrics = {
            "loss": loss.detach(),
            "masked_accuracy": correct.float() / masked_count.clamp_min(1),
            "mask_fraction": masked_count.float() / batch.response_positions.sum().clamp_min(1),
            "mean_t": batch.timesteps.mean(),
        }
    return loss, metrics
