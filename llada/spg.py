from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import LLaDA


@dataclass
class BlockwisePerturbations:
    noisy_tokens: torch.Tensor
    active_response_masks: torch.Tensor
    empirical_timesteps: torch.Tensor


def make_blockwise_perturbations(
    clean_tokens: torch.Tensor,
    *,
    prompt_length: int,
    completion_mask: torch.Tensor,
    mask_id: int,
    block_length: int,
    mc_samples: int,
    prompt_mask_probability: float,
) -> BlockwisePerturbations:
    """SPG block-wise masking: clean past, partial current block, masked future."""
    batch_size, total_length = clean_tokens.shape
    generation_length = total_length - prompt_length
    if generation_length % block_length != 0:
        raise ValueError("generation length must be divisible by block_length")
    if completion_mask.shape != (batch_size, generation_length):
        raise ValueError("completion_mask has an incompatible shape")
    noisy = clean_tokens[:, None, :].repeat(1, mc_samples, 1)
    active = torch.zeros(
        (batch_size, mc_samples, generation_length),
        dtype=torch.bool,
        device=clean_tokens.device,
    )

    for row in range(batch_size):
        valid_length = int(completion_mask[row].sum().item())
        valid_length = max(1, valid_length)
        valid_blocks = (valid_length + block_length - 1) // block_length
        for sample in range(mc_samples):
            block = int(torch.randint(valid_blocks, (1,), device=clean_tokens.device).item())
            start = block * block_length
            end = min(start + block_length, valid_length)
            block_size = end - start
            mask_count = int(torch.randint(1, block_size + 1, (1,), device=clean_tokens.device).item())
            selected = torch.randperm(block_size, device=clean_tokens.device)[:mask_count] + start
            response_mask = torch.zeros(generation_length, dtype=torch.bool, device=clean_tokens.device)
            response_mask[selected] = True
            response_mask[end:] = True

            # Light perturbation of prompt and already-visible response context.
            prompt_random = torch.rand(prompt_length, device=clean_tokens.device)
            noisy[row, sample, :prompt_length].masked_fill_(
                prompt_random < prompt_mask_probability, mask_id
            )
            visible_response = torch.arange(generation_length, device=clean_tokens.device) < end
            context_random = torch.rand(generation_length, device=clean_tokens.device)
            context_perturb = (
                visible_response
                & ~response_mask
                & completion_mask[row]
                & (context_random < prompt_mask_probability)
            )
            response_mask |= context_perturb
            noisy[row, sample, prompt_length:].masked_fill_(response_mask, mask_id)
            active[row, sample] = response_mask & completion_mask[row] & visible_response

    empirical_t = active.sum(dim=-1).float() / completion_mask.sum(dim=-1)[:, None].clamp_min(1)
    return BlockwisePerturbations(noisy, active, empirical_t.clamp_min(1e-8))


def spg_scores(
    model: LLaDA,
    reference_model: LLaDA,
    clean_tokens: torch.Tensor,
    perturbations: BlockwisePerturbations,
    *,
    prompt_length: int,
    completion_mask: torch.Tensor,
    eubo_beta: float,
    mixture_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate per-sequence ELBO, EUBO, mixed negative score, and reference KL."""
    batch_size, mc_samples, total_length = perturbations.noisy_tokens.shape
    generation_length = total_length - prompt_length
    flat_noisy = perturbations.noisy_tokens.reshape(batch_size * mc_samples, total_length)
    logits = model(flat_noisy).view(batch_size, mc_samples, total_length, -1)
    response_logits = logits[:, :, prompt_length:]
    targets = clean_tokens[:, None, prompt_length:].expand(-1, mc_samples, -1)
    log_probs = F.log_softmax(response_logits.float(), dim=-1)
    target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    target_probs = target_log_probs.exp()
    active = perturbations.active_response_masks

    per_mc_elbo = (target_log_probs * active).sum(dim=-1) / active.sum(dim=-1).clamp_min(1)
    elbo = per_mc_elbo.mean(dim=1)

    weighted_probability = (
        target_probs.pow(eubo_beta)
        * active
        / perturbations.empirical_timesteps[:, :, None]
    ).mean(dim=1)
    eubo_mask = weighted_probability.gt(0) & completion_mask
    eubo = (
        weighted_probability.clamp_min(1e-12).log().div(eubo_beta) * eubo_mask
    ).sum(dim=-1) / eubo_mask.sum(dim=-1).clamp_min(1)
    mixed = mixture_weight * eubo + (1.0 - mixture_weight) * elbo

    with torch.no_grad():
        reference_logits = reference_model(flat_noisy).view(
            batch_size, mc_samples, total_length, -1
        )[:, :, prompt_length:]
        reference_log_probs = F.log_softmax(reference_logits.float(), dim=-1)
    probabilities = log_probs.exp()
    token_kl = (probabilities * (log_probs - reference_log_probs)).sum(dim=-1)
    kl = (token_kl * active).sum() / active.sum().clamp_min(1)
    if elbo.shape != (batch_size,) or eubo.shape != (batch_size,):
        raise RuntimeError("unexpected SPG score shape")
    return elbo, eubo, mixed, kl
