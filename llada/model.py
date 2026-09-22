from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LLaDAConfig:
    vocab_size: int
    max_seq_len: int = 256
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    ffn_dim: int = 768
    dropout: float = 0.0
    rope_base: float = 10_000.0

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("the attention head dimension must be even for RoPE")

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.to(dtype=x.dtype) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        angles = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", angles.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None, :, :], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        cos = self.cos[:, :, :seq_len].to(dtype=q.dtype)
        sin = self.sin[:, :, :seq_len].to(dtype=q.dtype)
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class BidirectionalAttention(nn.Module):
    def __init__(self, config: LLaDAConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = config.dropout
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)

        # LLaDA is a masked diffusion model: every position may attend both left and right.
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, width)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, config: LLaDAConfig):
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.ffn_dim, bias=False)
        self.up = nn.Linear(config.n_embd, config.ffn_dim, bias=False)
        self.down = nn.Linear(config.ffn_dim, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: LLaDAConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = BidirectionalAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.ffn = SwiGLU(config)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.resid_dropout(self.attn(self.attn_norm(x)))
        x = x + self.resid_dropout(self.ffn(self.ffn_norm(x)))
        return x


class LLaDA(nn.Module):
    """LLaMA-style, non-causal Transformer used as LLaDA's mask predictor."""

    def __init__(self, config: LLaDAConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        self._scale_residual_projections()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        scale = 0.02 / math.sqrt(2 * self.config.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=scale)
            nn.init.normal_(block.ffn.down.weight, mean=0.0, std=scale)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError(f"sequence length exceeds max_seq_len={self.config.max_seq_len}")
        x = self.dropout(self.token_embedding(input_ids))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def resize_token_embeddings(model: LLaDA, new_vocab_size: int) -> LLaDA:
    """Return a model with an expanded vocabulary while preserving pretrained weights."""
    if new_vocab_size < model.config.vocab_size:
        raise ValueError("vocabulary shrinking is not supported")
    if new_vocab_size == model.config.vocab_size:
        return model
    config_values = model.config.to_dict()
    old_vocab_size = config_values["vocab_size"]
    config_values["vocab_size"] = new_vocab_size
    first_parameter = next(model.parameters())
    resized = LLaDA(LLaDAConfig(**config_values)).to(
        device=first_parameter.device, dtype=first_parameter.dtype
    )
    old_state = model.state_dict()
    new_state = resized.state_dict()
    with torch.no_grad():
        for name, old_value in old_state.items():
            new_value = new_state[name]
            if old_value.shape == new_value.shape:
                new_value.copy_(old_value)
            elif name in {"token_embedding.weight", "lm_head.weight"}:
                new_value[:old_vocab_size].copy_(old_value)
            else:
                raise ValueError(f"cannot resize parameter {name}: {old_value.shape} -> {new_value.shape}")
    resized.load_state_dict(new_state)
    return resized
