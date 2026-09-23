from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from llada.data import CharTokenizer, download_tiny_shakespeare
from llada.model import LLaDA, LLaDAConfig
from llada.reward import CharacterNGramScorer, RewardBreakdown, shakespeare_reward
from llada.sampling import generate_blockwise
from llada.sft_data import ShakespeareSFTData
from llada.spg import make_blockwise_perturbations, spg_scores


@dataclass
class SPGConfig:
    checkpoint: str = "out/shakespeare-sft-3epoch/best.pt"
    data_dir: str = "data/shakespeare"
    out_dir: str = "out/shakespeare-spg"
    max_updates: int = 200
    group_size: int = 4
    generation_length: int = 96
    diffusion_steps: int = 32
    block_length: int = 16
    mc_samples: int = 2
    inner_updates: int = 2
    eubo_beta: float = 1.5
    mixture_weight: float = 0.5
    learning_rate: float = 1e-6
    warmup_updates: int = 10
    weight_decay: float = 0.1
    grad_clip: float = 0.2
    temperature: float = 0.9
    top_k: int = 20
    prompt_mask_probability: float = 0.15
    kl_coefficient: float = 0.02
    context_turns: int = 2
    max_response_chars: int = 128
    eval_interval: int = 25
    eval_prompts: int = 4
    checkpoint_interval: int = 25
    histogram_interval: int = 50
    text_interval: int = 10
    seed: int = 1337
    device: str = "auto"
    tensorboard: bool = True
    resume: str = ""


def parse_args() -> SPGConfig:
    parser = argparse.ArgumentParser(description="SPG reinforcement learning for Shakespeare LLaDA")
    for name, field in SPGConfig.__dataclass_fields__.items():
        default = field.default
        argument = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(argument, action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(argument, type=type(default), default=default)
    return SPGConfig(**vars(parser.parse_args()))


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def completion_mask(response_tokens: torch.Tensor, eos_id: int) -> torch.Tensor:
    positions = torch.arange(response_tokens.size(1), device=response_tokens.device)[None, :]
    eos_matches = response_tokens.eq(eos_id)
    has_eos = eos_matches.any(dim=1)
    first_eos = eos_matches.float().argmax(dim=1).long()
    lengths = torch.where(
        has_eos, first_eos + 1, torch.full_like(first_eos, response_tokens.size(1))
    )
    return positions < lengths[:, None]


def sample_pair(
    data: ShakespeareSFTData,
    split: str,
    *,
    maximum_prompt_length: int,
) -> tuple[list[int], list[int]]:
    pairs = data.train if split == "train" else data.val
    index = int(torch.randint(len(pairs), (1,)).item())
    prompt, response = pairs[index]
    return prompt[-maximum_prompt_length:], response


def score_responses(
    response_tokens: torch.Tensor,
    reference_text: str,
    tokenizer: CharTokenizer,
    scorer: CharacterNGramScorer,
    generation_length: int,
) -> tuple[torch.Tensor, list[RewardBreakdown], list[str]]:
    breakdowns, texts = [], []
    for row in response_tokens.tolist():
        has_eos = tokenizer.eos_id in row
        text = tokenizer.decode(row)
        texts.append(text)
        breakdowns.append(
            shakespeare_reward(
                text,
                reference_text,
                fluency_scorer=scorer,
                has_eos=has_eos,
                generation_length=generation_length,
            )
        )
    rewards = torch.tensor(
        [item.total for item in breakdowns], dtype=torch.float32, device=response_tokens.device
    )
    return rewards, breakdowns, texts


def rollout_group(
    model: LLaDA,
    prompt: list[int],
    reference: list[int],
    tokenizer: CharTokenizer,
    scorer: CharacterNGramScorer,
    config: SPGConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[RewardBreakdown], list[str], str]:
    prompt_tensor = torch.tensor(prompt, dtype=torch.long, device=device)[None, :]
    prompt_tensor = prompt_tensor.expand(config.group_size, -1).clone()
    generated = generate_blockwise(
        model,
        mask_id=tokenizer.mask_id,
        prompt_tokens=prompt_tensor,
        generation_length=config.generation_length,
        steps=config.diffusion_steps,
        block_length=config.block_length,
        temperature=config.temperature,
        top_k=config.top_k,
    ).clone()
    response_tokens = generated[:, len(prompt) :]
    reference_text = tokenizer.decode(reference)[: config.generation_length]
    rewards, breakdowns, texts = score_responses(
        response_tokens, reference_text, tokenizer, scorer, config.generation_length
    )
    return generated, rewards, breakdowns, texts, reference_text


@torch.no_grad()
def evaluate_policy(
    model: LLaDA,
    data: ShakespeareSFTData,
    tokenizer: CharTokenizer,
    scorer: CharacterNGramScorer,
    config: SPGConfig,
    device: torch.device,
) -> dict[str, float]:
    totals = {
        "reward": 0.0,
        "reference_similarity": 0.0,
        "fluency": 0.0,
        "eos_length": 0.0,
        "formatting": 0.0,
        "anti_repetition": 0.0,
        "eos_rate": 0.0,
        "response_length": 0.0,
    }
    maximum_prompt_length = model.config.max_seq_len - config.generation_length
    count = 0
    for _ in range(config.eval_prompts):
        prompt, reference = sample_pair(
            data, "val", maximum_prompt_length=maximum_prompt_length
        )
        _, _, breakdowns, _, _ = rollout_group(
            model, prompt, reference, tokenizer, scorer, config, device
        )
        for item in breakdowns:
            totals["reward"] += item.total
            totals["reference_similarity"] += item.reference_similarity
            totals["fluency"] += item.fluency
            totals["eos_length"] += item.eos_length
            totals["formatting"] += item.formatting
            totals["anti_repetition"] += item.anti_repetition
            totals["eos_rate"] += float(item.has_eos)
            totals["response_length"] += item.response_length
            count += 1
    return {name: value / max(1, count) for name, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: LLaDA,
    optimizer: torch.optim.Optimizer,
    update: int,
    best_eval_reward: float,
    config: SPGConfig,
    tokenizer: CharTokenizer,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": update,
            "iteration": update,
            "best_eval_reward": best_eval_reward,
            "model_config": model.config.to_dict(),
            "spg_config": asdict(config),
            "tokenizer_chars": tokenizer.chars,
            "tokenizer_has_eos": tokenizer.has_eos,
            "reference_checkpoint": config.checkpoint,
        },
        path,
    )


def mean_component(items: list[RewardBreakdown], name: str) -> float:
    return sum(float(getattr(item, name)) for item in items) / len(items)


def main() -> None:
    config = parse_args()
    if config.generation_length % config.block_length != 0:
        raise ValueError("--generation-length must be divisible by --block-length")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = select_device(config.device)
    base_checkpoint = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
    if not base_checkpoint.get("tokenizer_has_eos", False):
        raise ValueError("SPG requires an EOS-aware SFT checkpoint")
    model_config = LLaDAConfig(**base_checkpoint["model_config"])
    model = LLaDA(model_config)
    model.load_state_dict(base_checkpoint["model"])
    reference_model = LLaDA(model_config)
    reference_model.load_state_dict(base_checkpoint["model"])
    model = model.to(device)
    reference_model = reference_model.to(device).eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    tokenizer = CharTokenizer(
        tuple(base_checkpoint["tokenizer_chars"]), has_eos=True
    )
    if config.generation_length > model.config.max_seq_len:
        raise ValueError("generation length exceeds the model context window")

    data = ShakespeareSFTData(
        config.data_dir,
        tokenizer,
        max_seq_len=model.config.max_seq_len,
        context_turns=config.context_turns,
        max_response_chars=config.max_response_chars,
    )
    corpus = download_tiny_shakespeare(Path(config.data_dir)).read_text(encoding="utf-8")
    heldout_text = corpus[int(len(corpus) * 0.9) :]
    fluency_scorer = CharacterNGramScorer(heldout_text, order=4)

    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(0.9, 0.99),
    )
    start_update, best_eval_reward = 0, -math.inf
    if config.resume:
        resumed = torch.load(config.resume, map_location="cpu", weights_only=False)
        if resumed["model_config"] != model_config.to_dict():
            raise ValueError("resume checkpoint model configuration does not match")
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        start_update = resumed["update"] + 1
        best_eval_reward = resumed.get("best_eval_reward", -math.inf)

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_dir / "tokenizer.json")
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    writer = None
    if config.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(out_dir / "tensorboard"), purge_step=start_update)
        writer.add_text(
            "run/spg_config", f"```json\n{json.dumps(asdict(config), indent=2)}\n```", start_update
        )

    print(
        f"device={device} parameters={model.parameter_count() / 1e6:.2f}M "
        f"train_pairs={len(data.train):,} group={config.group_size} "
        f"generation={config.generation_length} steps={config.diffusion_steps}"
    )
    maximum_prompt_length = model.config.max_seq_len - config.generation_length
    model.train()
    try:
        for update in range(start_update, config.max_updates):
            started = time.perf_counter()
            learning_rate = config.learning_rate * min(
                1.0, (update + 1) / max(1, config.warmup_updates)
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            prompt, reference = sample_pair(
                data, "train", maximum_prompt_length=maximum_prompt_length
            )
            generated, rewards, breakdowns, texts, reference_text = rollout_group(
                model, prompt, reference, tokenizer, fluency_scorer, config, device
            )
            advantages = rewards - rewards.mean()
            response_tokens = generated[:, len(prompt) :]
            valid_completion = completion_mask(response_tokens, tokenizer.eos_id)

            inner_metrics = []
            for _ in range(config.inner_updates):
                perturbations = make_blockwise_perturbations(
                    generated,
                    prompt_length=len(prompt),
                    completion_mask=valid_completion,
                    mask_id=tokenizer.mask_id,
                    block_length=config.block_length,
                    mc_samples=config.mc_samples,
                    prompt_mask_probability=config.prompt_mask_probability,
                )
                optimizer.zero_grad(set_to_none=True)
                elbo, eubo, mixed, kl = spg_scores(
                    model,
                    reference_model,
                    generated,
                    perturbations,
                    prompt_length=len(prompt),
                    completion_mask=valid_completion,
                    eubo_beta=config.eubo_beta,
                    mixture_weight=config.mixture_weight,
                )
                selected_scores = torch.where(advantages >= 0, elbo, mixed)
                policy_objective = (advantages.detach() * selected_scores).mean()
                loss = -policy_objective + config.kl_coefficient * kl
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                inner_metrics.append(
                    {
                        "loss": loss.item(),
                        "policy_objective": policy_objective.item(),
                        "elbo": elbo.mean().item(),
                        "eubo": eubo.mean().item(),
                        "mixed": mixed.mean().item(),
                        "kl": kl.item(),
                        "grad_norm": float(grad_norm),
                    }
                )

            elapsed = time.perf_counter() - started
            aggregate = {
                key: sum(item[key] for item in inner_metrics) / len(inner_metrics)
                for key in inner_metrics[0]
            }
            unique_fraction = len(set(texts)) / len(texts)
            print(
                f"update {update:4d} | reward {rewards.mean().item():.4f} +/- "
                f"{rewards.std(unbiased=False).item():.4f} | loss {aggregate['loss']:.4f} | "
                f"KL {aggregate['kl']:.5f} | eos {mean_component(breakdowns, 'has_eos'):.2f} | "
                f"{elapsed:.2f}s"
            )

            if writer is not None:
                scalars = {
                    "reward/train_mean": rewards.mean().item(),
                    "reward/train_min": rewards.min().item(),
                    "reward/train_max": rewards.max().item(),
                    "reward/train_std": rewards.std(unbiased=False).item(),
                    "reward/reference_similarity": mean_component(breakdowns, "reference_similarity"),
                    "reward/fluency": mean_component(breakdowns, "fluency"),
                    "reward/eos_length": mean_component(breakdowns, "eos_length"),
                    "reward/formatting": mean_component(breakdowns, "formatting"),
                    "reward/anti_repetition": mean_component(breakdowns, "anti_repetition"),
                    "advantage/std": advantages.std(unbiased=False).item(),
                    "advantage/positive_fraction": advantages.gt(0).float().mean().item(),
                    "advantage/negative_fraction": advantages.lt(0).float().mean().item(),
                    "objective/loss": aggregate["loss"],
                    "objective/policy": aggregate["policy_objective"],
                    "objective/elbo": aggregate["elbo"],
                    "objective/eubo": aggregate["eubo"],
                    "objective/negative_mixture": aggregate["mixed"],
                    "objective/reference_kl": aggregate["kl"],
                    "optimization/learning_rate": learning_rate,
                    "optimization/gradient_norm": aggregate["grad_norm"],
                    "rollout/eos_rate": mean_component(breakdowns, "has_eos"),
                    "rollout/response_length": mean_component(breakdowns, "response_length"),
                    "rollout/unique_fraction": unique_fraction,
                    "performance/update_seconds": elapsed,
                    "performance/generated_chars_per_second": (
                        config.group_size * config.generation_length / max(elapsed, 1e-9)
                    ),
                }
                for name, value in scalars.items():
                    writer.add_scalar(name, value, update)
                if update % config.text_interval == 0:
                    prompt_text = tokenizer.decode(prompt)
                    body = (
                        f"**Prompt**\n```text\n{prompt_text}\n```\n"
                        f"**Reference**\n```text\n{reference_text}\n```\n"
                        f"**Sample**\n```text\n{texts[0]}\n```\n"
                        f"Reward: `{breakdowns[0].total:.4f}`"
                    )
                    writer.add_text("samples/train", body, update)
                if config.histogram_interval > 0 and (update + 1) % config.histogram_interval == 0:
                    for name, parameter in model.named_parameters():
                        writer.add_histogram(
                            f"parameters/{name}", parameter.detach().float().cpu(), update
                        )

            should_evaluate = update % config.eval_interval == 0 or update == config.max_updates - 1
            if should_evaluate:
                evaluation = evaluate_policy(
                    model, data, tokenizer, fluency_scorer, config, device
                )
                print(
                    f"evaluation  | reward {evaluation['reward']:.4f} | "
                    f"similarity {evaluation['reference_similarity']:.3f} | "
                    f"eos {evaluation['eos_rate']:.3f}"
                )
                if writer is not None:
                    for name, value in evaluation.items():
                        writer.add_scalar(f"evaluation/{name}", value, update)
                    writer.flush()
                if evaluation["reward"] > best_eval_reward:
                    best_eval_reward = evaluation["reward"]
                    save_checkpoint(
                        out_dir / "best.pt",
                        model,
                        optimizer,
                        update,
                        best_eval_reward,
                        config,
                        tokenizer,
                    )

            if (update + 1) % config.checkpoint_interval == 0 or update == config.max_updates - 1:
                save_checkpoint(
                    out_dir / "latest.pt",
                    model,
                    optimizer,
                    update,
                    best_eval_reward,
                    config,
                    tokenizer,
                )
    finally:
        if writer is not None:
            writer.flush()
            writer.close()


if __name__ == "__main__":
    main()
