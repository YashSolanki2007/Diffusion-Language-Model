from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


@dataclass(frozen=True)
class CharTokenizer:
    chars: tuple[str, ...]
    has_eos: bool = False

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(tuple(sorted(set(text))))

    @property
    def mask_id(self) -> int:
        return len(self.chars)

    @property
    def eos_id(self) -> int:
        if not self.has_eos:
            raise ValueError("this tokenizer does not define an EOS token")
        return len(self.chars) + 1

    @property
    def vocab_size(self) -> int:
        return len(self.chars) + 1 + int(self.has_eos)

    def encode(self, text: str) -> list[int]:
        lookup = {char: index for index, char in enumerate(self.chars)}
        unknown = sorted(set(text) - set(lookup))
        if unknown:
            rendered = ", ".join(repr(char) for char in unknown)
            raise ValueError(f"text contains characters outside the training vocabulary: {rendered}")
        return [lookup[char] for char in text]

    def decode(self, token_ids: list[int], *, stop_at_eos: bool = True) -> str:
        pieces = []
        for token in token_ids:
            if self.has_eos and token == self.eos_id:
                if stop_at_eos:
                    break
                pieces.append("<EOS>")
            elif token == self.mask_id:
                pieces.append("<MASK>")
            else:
                pieces.append(self.chars[token])
        return "".join(pieces)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"chars": self.chars, "has_eos": self.has_eos}, ensure_ascii=False, indent=2)
            + "\n"
        )

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        payload = json.loads(path.read_text())
        return cls(tuple(payload["chars"]), payload.get("has_eos", False))


def download_tiny_shakespeare(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / "input.txt"
    if destination.exists():
        return destination
    temporary = destination.with_suffix(".txt.download")
    print(f"downloading Tiny Shakespeare to {destination}")
    try:
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


class ShakespeareData:
    def __init__(self, data_dir: str | Path, train_fraction: float = 0.9):
        path = download_tiny_shakespeare(Path(data_dir))
        text = path.read_text(encoding="utf-8")
        self.tokenizer = CharTokenizer.from_text(text)
        tokens = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        split = int(len(tokens) * train_fraction)
        self.train = tokens[:split]
        self.val = tokens[split:]

    def get_batch(
        self,
        split: str,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        data = self.train if split == "train" else self.val
        if seq_len >= len(data):
            raise ValueError(f"seq_len={seq_len} is too large for the {split} split")
        starts = torch.randint(0, len(data) - seq_len, (batch_size,), generator=generator)
        batch = torch.stack([data[start : start + seq_len] for start in starts.tolist()])
        return batch.to(device, non_blocking=True)
