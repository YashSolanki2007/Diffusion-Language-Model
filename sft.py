from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from llada.data import CharTokenizer
from llada.model import LLaDA, LLaDAConfig, resize_token_embeddings
from llada.objective import llada_sft_loss, make_sft_diffusion_batch
from llada.sft_data import ShakespeareSFTData


@dataclass
class SFTConfig:
    pretrained_checkpoint: str = "out/shakespeare/best.pt"
    data_dir: str = "data/shakespeare"
    out_dir: str = "out/shakespeare-sft"
    max_iters: int = 500
    batch_size: int = 8
    context_turns: int = 2
    max_response_chars: int = 128
    learning_rate: float = 2.5e-5
    min_lr: float = 2.5e-6
    warmup_iters: int = 50
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 50
    eval_iters: int = 10
    log_interval: int = 10
    checkpoint_interval: int = 100
    seed: int = 1337
    device: str = "auto"
    tensorboard: bool = True


def parse_args() -> SFTConfig:
    parser = argparse.ArgumentParser(description="SFT a pretrained LLaDA on Shakespeare dialogue")
    for name, field in SFTConfig.__dataclass_fields__.items():
        default = field.default
        argument = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(argument, action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(argument, type=type(default), default=default)
    return SFTConfig(**vars(parser.parse_args()))


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def learning_rate_for(iteration: int, config: SFTConfig) -> float:
    if iteration < config.warmup_iters:
        return config.learning_rate * (iteration + 1) / max(1, config.warmup_iters)
    decay_start = math.floor(config.max_iters * 0.9)
    if iteration < decay_start:
        return config.learning_rate
    ratio = (iteration - decay_start) / max(1, config.max_iters - decay_start)
    return config.learning_rate + ratio * (config.min_lr - config.learning_rate)


def save_checkpoint(
    path: Path,
    model: LLaDA,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    best_val_loss: float,
    config: SFTConfig,
    tokenizer: CharTokenizer,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "best_val_loss": best_val_loss,
            "model_config": model.config.to_dict(),
            "sft_config": asdict(config),
            "tokenizer_chars": tokenizer.chars,
            "tokenizer_has_eos": True,
        },
        path,
    )


@torch.no_grad()
def evaluate(
    model: LLaDA,
    data: ShakespeareSFTData,
    config: SFTConfig,
    device: torch.device,
    autocast_context,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "masked_accuracy": 0.0, "mask_fraction": 0.0}
    for _ in range(config.eval_iters):
        clean, response_positions = data.get_batch("val", config.batch_size, device)
        batch = make_sft_diffusion_batch(clean, response_positions, data.tokenizer.mask_id)
        with autocast_context():
            _, metrics = llada_sft_loss(model(batch.noisy_tokens), batch)
        for key in totals:
            totals[key] += metrics[key].item()
    model.train()
    return {key: value / config.eval_iters for key, value in totals.items()}


def main() -> None:
    config = parse_args()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = select_device(config.device)
    checkpoint = torch.load(config.pretrained_checkpoint, map_location="cpu", weights_only=False)
    base_model = LLaDA(LLaDAConfig(**checkpoint["model_config"]))
    base_model.load_state_dict(checkpoint["model"])
    tokenizer = CharTokenizer(tuple(checkpoint["tokenizer_chars"]), has_eos=True)
    model = resize_token_embeddings(base_model, tokenizer.vocab_size).to(device)
    data = ShakespeareSFTData(
        config.data_dir,
        tokenizer,
        max_seq_len=model.config.max_seq_len,
        context_turns=config.context_turns,
        max_response_chars=config.max_response_chars,
    )
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_dir / "tokenizer.json")
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
    )
    autocast_context = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if device.type == "cuda"
        else nullcontext
    )
    writer = None
    if config.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(out_dir / "tensorboard"))
        writer.add_text("run/sft_config", f"```json\n{json.dumps(asdict(config), indent=2)}\n```", 0)

    print(
        f"device={device} parameters={model.parameter_count() / 1e6:.2f}M "
        f"pairs={len(data.train):,}/{len(data.val):,} train/val vocab={tokenizer.vocab_size}"
    )
    best_val_loss = math.inf
    tick = time.perf_counter()
    model.train()
    try:
        for iteration in range(config.max_iters):
            lr = learning_rate_for(iteration, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            clean, response_positions = data.get_batch("train", config.batch_size, device)
            batch = make_sft_diffusion_batch(clean, response_positions, tokenizer.mask_id)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                loss, metrics = llada_sft_loss(model(batch.noisy_tokens), batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            if writer is not None:
                writer.add_scalar("train/loss", metrics["loss"].item(), iteration)
                writer.add_scalar("train/masked_accuracy", metrics["masked_accuracy"].item(), iteration)
                writer.add_scalar("train/mask_fraction", metrics["mask_fraction"].item(), iteration)
                writer.add_scalar("train/mean_t", metrics["mean_t"].item(), iteration)
                writer.add_scalar("optimization/learning_rate", lr, iteration)
                writer.add_scalar("optimization/gradient_norm", float(grad_norm), iteration)

            if iteration % config.log_interval == 0 or iteration == config.max_iters - 1:
                elapsed = time.perf_counter() - tick
                tick = time.perf_counter()
                print(
                    f"iter {iteration:5d} | loss {metrics['loss'].item():.4f} | "
                    f"acc {metrics['masked_accuracy'].item():.3f} | "
                    f"masked {metrics['mask_fraction'].item():.3f} | "
                    f"lr {lr:.2e} | grad {float(grad_norm):.2f} | {elapsed:.2f}s"
                )

            if iteration % config.eval_interval == 0 or iteration == config.max_iters - 1:
                val = evaluate(model, data, config, device, autocast_context)
                print(f"validation  | loss {val['loss']:.4f} | acc {val['masked_accuracy']:.3f}")
                if writer is not None:
                    for key, value in val.items():
                        writer.add_scalar(f"validation/{key}", value, iteration)
                    writer.flush()
                if val["loss"] < best_val_loss:
                    best_val_loss = val["loss"]
                    save_checkpoint(
                        out_dir / "best.pt", model, optimizer, iteration,
                        best_val_loss, config, tokenizer,
                    )

            if (iteration + 1) % config.checkpoint_interval == 0 or iteration == config.max_iters - 1:
                save_checkpoint(
                    out_dir / "latest.pt", model, optimizer, iteration,
                    best_val_loss, config, tokenizer,
                )
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
