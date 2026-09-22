from __future__ import annotations

import argparse
from pathlib import Path

import torch

from llada.data import CharTokenizer
from llada.model import LLaDA, LLaDAConfig
from llada.sampling import generate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a pretrained Shakespeare LLaDA")
    parser.add_argument("--checkpoint", default="out/shakespeare/best.pt")
    parser.add_argument("--prompt", default="", help="optional fixed text prefix")
    parser.add_argument("--length", type=int, default=256, help="number of new characters")
    parser.add_argument("--steps", type=int, default=128, help="reverse diffusion steps")
    parser.add_argument(
        "--remasking",
        choices=("low_confidence", "random"),
        default="low_confidence",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 uses the paper's greedy decoding; positive values sample",
    )
    parser.add_argument("--top-k", type=int, default=0, help="0 disables top-k filtering")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be at least 1")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint_path}. Run train.py first or pass --checkpoint."
        )

    torch.manual_seed(args.seed)
    device = select_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = LLaDAConfig(**checkpoint["model_config"])
    model = LLaDA(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = CharTokenizer(
        tuple(checkpoint["tokenizer_chars"]), checkpoint.get("tokenizer_has_eos", False)
    )
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError("checkpoint tokenizer and model vocabulary sizes do not match")
    encoded_prompt = tokenizer.encode(args.prompt)
    prompt_tokens = torch.tensor(encoded_prompt, dtype=torch.long, device=device)
    prompt_tokens = prompt_tokens.unsqueeze(0).expand(args.num_samples, -1).clone()

    samples = generate(
        model,
        mask_id=tokenizer.mask_id,
        prompt_tokens=prompt_tokens,
        generation_length=args.length,
        steps=args.steps,
        remasking=args.remasking,
        temperature=args.temperature,
        top_k=args.top_k,
        show_progress=args.show_progress,
    )
    print(
        f"checkpoint={checkpoint_path} iteration={checkpoint.get('iteration', 'unknown')} "
        f"device={device} strategy={args.remasking} steps={args.steps}"
    )
    for index, sample in enumerate(samples.tolist(), start=1):
        if args.num_samples > 1:
            print(f"\n--- sample {index} ---")
        print(tokenizer.decode(sample))


if __name__ == "__main__":
    main()
