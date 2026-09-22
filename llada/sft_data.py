from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .data import CharTokenizer, download_tiny_shakespeare


@dataclass(frozen=True)
class DialoguePair:
    prompt: str
    response: str


def parse_dialogue_pairs(
    text: str,
    *,
    context_turns: int = 2,
    max_response_chars: int = 128,
) -> list[DialoguePair]:
    """Turn speaker-labelled Shakespeare blocks into multi-turn SFT examples."""
    turns: list[tuple[str, str]] = []
    for block in text.split("\n\n"):
        lines = [line.rstrip() for line in block.strip().splitlines()]
        if len(lines) < 2 or not lines[0].endswith(":") or len(lines[0]) > 64:
            continue
        speaker = lines[0]
        response = "\n".join(lines[1:]).strip()
        if response:
            turns.append((speaker, response))

    pairs: list[DialoguePair] = []
    for index, (speaker, response) in enumerate(turns):
        history = turns[max(0, index - context_turns) : index]
        history_text = "".join(f"{name}\n{utterance}\n\n" for name, utterance in history)
        prompt = f"{history_text}{speaker}\n"
        pairs.append(DialoguePair(prompt=prompt, response=response[:max_response_chars]))
    return pairs


class ShakespeareSFTData:
    def __init__(
        self,
        data_dir: str | Path,
        tokenizer: CharTokenizer,
        *,
        max_seq_len: int,
        context_turns: int = 2,
        max_response_chars: int = 128,
        train_fraction: float = 0.9,
    ):
        if not tokenizer.has_eos:
            raise ValueError("SFT requires a tokenizer with EOS")
        text = download_tiny_shakespeare(Path(data_dir)).read_text(encoding="utf-8")
        pairs = parse_dialogue_pairs(
            text,
            context_turns=context_turns,
            max_response_chars=max_response_chars,
        )
        encoded = []
        for pair in pairs:
            response = tokenizer.encode(pair.response)
            max_prompt_len = max_seq_len - len(response) - 1
            if max_prompt_len < 1:
                continue
            prompt = tokenizer.encode(pair.prompt)[-max_prompt_len:]
            encoded.append((prompt, response))
        split = int(len(encoded) * train_fraction)
        self.train = encoded[:split]
        self.val = encoded[split:]
        self._buckets = {
            "train": self._make_length_buckets(self.train),
            "val": self._make_length_buckets(self.val),
        }
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    @staticmethod
    def _make_length_buckets(pairs: list[tuple[list[int], list[int]]]) -> dict[int, list[int]]:
        buckets: dict[int, list[int]] = {}
        for index, (prompt, response) in enumerate(pairs):
            bucket = (len(prompt) + len(response) + 1) // 32
            buckets.setdefault(bucket, []).append(index)
        return buckets

    def get_batch(
        self,
        split: str,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pairs = self.train if split == "train" else self.val
        # Choose an anchor uniformly from the data, then sample within its length bucket.
        # This retains an approximately uniform pair distribution while avoiding excessive EOS padding.
        anchor = torch.randint(0, len(pairs), (1,)).item()
        anchor_prompt, anchor_response = pairs[anchor]
        bucket_key = (len(anchor_prompt) + len(anchor_response) + 1) // 32
        bucket = self._buckets[split][bucket_key]
        choices = torch.randint(0, len(bucket), (batch_size,)).tolist()
        indices = [bucket[choice] for choice in choices]
        selected = [pairs[index] for index in indices]
        max_total = max(len(prompt) + len(response) + 1 for prompt, response in selected)
        clean_rows, response_rows = [], []
        for prompt, response in selected:
            eos_count = max_total - len(prompt) - len(response)
            clean_rows.append(prompt + response + [self.tokenizer.eos_id] * eos_count)
            response_rows.append([False] * len(prompt) + [True] * (len(response) + eos_count))
        return (
            torch.tensor(clean_rows, dtype=torch.long, device=device),
            torch.tensor(response_rows, dtype=torch.bool, device=device),
        )
