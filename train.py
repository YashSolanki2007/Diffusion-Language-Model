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

from llada.data import ShakespeareData
from llada.model import LLaDA, LLaDAConfig
from llada.objective import llada_pretraining_loss, make_diffusion_batch


@dataclass
class TrainConfig:
    data_dir: str = "data/shakespeare"
    out_dir: str = "out/shakespeare"
    max_iters: int = 5_000
    batch_size: int = 32
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    ffn_dim: int = 768
    dropout: float = 0.0
    learning_rate: float = 4e-4
    min_lr: float = 4e-5
    warmup_iters: int = 200
    decay_start_iter: int = 4_000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 25
    log_interval: int = 10
    checkpoint_interval: int = 500
    histogram_interval: int = 500
    variable_length_prob: float = 0.01
    tensorboard: bool = True
    tensorboard_log_dir: str = ""
    seed: int = 1337
    device: str = "auto"
    resume: str | None = None


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Pretrain a compact LLaDA on Tiny Shakespeare")
    for field_name, field in TrainConfig.__dataclass_fields__.items():
        default = field.default
        argument = "--" + field_name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(argument, action=argparse.BooleanOptionalAction, default=default)
        elif field_name == "resume":
            parser.add_argument(argument, type=str, default=None)
        else:
            parser.add_argument(argument, type=type(default), default=default)
    return TrainConfig(**vars(parser.parse_args()))


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def learning_rate_for(iteration: int, config: TrainConfig) -> float:
    if iteration < config.warmup_iters:
        return config.learning_rate * (iteration + 1) / max(1, config.warmup_iters)
    if iteration < config.decay_start_iter:
        return config.learning_rate
    if iteration >= config.max_iters:
        return config.min_lr
    ratio = (iteration - config.decay_start_iter) / max(1, config.max_iters - config.decay_start_iter)
    return config.learning_rate + ratio * (config.min_lr - config.learning_rate)


def make_optimizer(model: LLaDA, config: TrainConfig, device: torch.device) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    kwargs = {"lr": config.learning_rate, "betas": (0.9, 0.95)}
    if device.type == "cuda":
        kwargs["fused"] = True
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": config.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        **kwargs,
    )


def choose_sequence_length(config: TrainConfig) -> int:
    if random.random() < config.variable_length_prob:
        return random.randint(1, config.block_size)
    return config.block_size


@torch.no_grad()
def evaluate(
    model: LLaDA,
    dataset: ShakespeareData,
    config: TrainConfig,
    device: torch.device,
    autocast_context,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "masked_accuracy": 0.0, "mask_fraction": 0.0}
    for _ in range(config.eval_iters):
        clean = dataset.get_batch("val", config.batch_size, config.block_size, device)
        diffusion = make_diffusion_batch(clean, dataset.tokenizer.mask_id)
        with autocast_context():
            logits = model(diffusion.noisy_tokens)
            _, metrics = llada_pretraining_loss(logits, diffusion)
        for key in totals:
            totals[key] += metrics[key].item()
    model.train()
    return {key: value / config.eval_iters for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: LLaDA,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    best_val_loss: float,
    config: TrainConfig,
    tokenizer_chars: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "best_val_loss": best_val_loss,
            "model_config": model.config.to_dict(),
            "train_config": asdict(config),
            "tokenizer_chars": tokenizer_chars,
        },
        path,
    )


def main() -> None:
    config = parse_args()
    if config.decay_start_iter > config.max_iters:
        config.decay_start_iter = config.max_iters
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = select_device(config.device)
    dataset = ShakespeareData(config.data_dir)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset.tokenizer.save(out_dir / "tokenizer.json")
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    model_config = LLaDAConfig(
        vocab_size=dataset.tokenizer.vocab_size,
        max_seq_len=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
    )
    model = LLaDA(model_config).to(device)
    optimizer = make_optimizer(model, config, device)
    start_iter, best_val_loss = 0, math.inf
    if config.resume:
        checkpoint = torch.load(config.resume, map_location=device, weights_only=False)
        if checkpoint["model_config"] != model_config.to_dict():
            raise ValueError("checkpoint model configuration does not match CLI configuration")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iter = checkpoint["iteration"] + 1
        best_val_loss = checkpoint.get("best_val_loss", math.inf)

    writer = None
    if config.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError(
                "TensorBoard logging is enabled but tensorboard is not installed. "
                "Run `python3 -m pip install -r requirements.txt`, or pass --no-tensorboard."
            ) from error
        log_dir = Path(config.tensorboard_log_dir) if config.tensorboard_log_dir else out_dir / "tensorboard"
        writer = SummaryWriter(log_dir=str(log_dir), purge_step=start_iter)
        writer.add_text("run/train_config", f"```json\n{json.dumps(asdict(config), indent=2)}\n```", start_iter)
        writer.add_text(
            "run/model_config",
            f"```json\n{json.dumps(model_config.to_dict(), indent=2)}\n```",
            start_iter,
        )

    if device.type == "cuda":
        autocast_context = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_context = nullcontext

    print(
        f"device={device} parameters={model.parameter_count() / 1e6:.2f}M "
        f"vocab={model_config.vocab_size} train_chars={len(dataset.train):,}"
    )
    model.train()
    tick = time.perf_counter()
    tokens_since_log = 0
    try:
        for iteration in range(start_iter, config.max_iters):
            lr = learning_rate_for(iteration, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            seq_len = choose_sequence_length(config)
            tokens_since_log += config.batch_size * seq_len
            clean = dataset.get_batch("train", config.batch_size, seq_len, device)
            diffusion = make_diffusion_batch(clean, dataset.tokenizer.mask_id)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                logits = model(diffusion.noisy_tokens)
                loss, metrics = llada_pretraining_loss(logits, diffusion)
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
                writer.add_scalar("data/sequence_length", seq_len, iteration)

            should_log = iteration % config.log_interval == 0 or iteration == config.max_iters - 1
            if should_log:
                elapsed = time.perf_counter() - tick
                tokens_per_second = tokens_since_log / max(elapsed, 1e-9)
                tick = time.perf_counter()
                tokens_since_log = 0
                print(
                    f"iter {iteration:6d} | loss {metrics['loss'].item():.4f} | "
                    f"acc {metrics['masked_accuracy'].item():.3f} | "
                    f"masked {metrics['mask_fraction'].item():.3f} | "
                    f"lr {lr:.2e} | grad {float(grad_norm):.2f} | "
                    f"{tokens_per_second:,.0f} tok/s | {elapsed:.2f}s"
                )
                if writer is not None:
                    writer.add_scalar("performance/tokens_per_second", tokens_per_second, iteration)

            if (
                writer is not None
                and config.histogram_interval > 0
                and (iteration + 1) % config.histogram_interval == 0
            ):
                for name, parameter in model.named_parameters():
                    writer.add_histogram(f"parameters/{name}", parameter.detach().float().cpu(), iteration)

            should_eval = iteration % config.eval_interval == 0 or iteration == config.max_iters - 1
            if should_eval:
                val = evaluate(model, dataset, config, device, autocast_context)
                print(
                    f"validation    | loss {val['loss']:.4f} | "
                    f"acc {val['masked_accuracy']:.3f} | masked {val['mask_fraction']:.3f}"
                )
                if writer is not None:
                    writer.add_scalar("validation/loss", val["loss"], iteration)
                    writer.add_scalar("validation/masked_accuracy", val["masked_accuracy"], iteration)
                    writer.add_scalar("validation/mask_fraction", val["mask_fraction"], iteration)
                    writer.flush()
                if val["loss"] < best_val_loss:
                    best_val_loss = val["loss"]
                    save_checkpoint(
                        out_dir / "best.pt", model, optimizer, iteration, best_val_loss,
                        config, dataset.tokenizer.chars,
                    )

            if (iteration + 1) % config.checkpoint_interval == 0 or iteration == config.max_iters - 1:
                save_checkpoint(
                    out_dir / "latest.pt", model, optimizer, iteration, best_val_loss,
                    config, dataset.tokenizer.chars,
                )
    finally:
        if writer is not None:
            writer.flush()
            writer.close()


if __name__ == "__main__":
    main()
