"""Multi-head causal self-attention for the GPT baseline."""

import math

import torch
from torch import nn

from configs.model_config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Accept and return hidden states (B, T, H).

    Each position can attend only to itself and earlier positions.
    Inputs are dense sequences without padding, matching the current
    interface. Residual connections and normalization belong to the block.
    This explicit baseline uses O(T^2) attention memory.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        # 한 번의 선형 변환으로 Query, Key, Value를 함께 만든다.
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        for layer in (self.qkv, self.out_proj):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            nn.init.zeros_(layer.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape (B, T, H)")
        batch, length, hidden = hidden_states.shape
        if hidden != self.hidden_size or length == 0:
            raise ValueError("hidden size must match config and T must be positive")
        if not hidden_states.is_floating_point():
            raise TypeError("hidden_states must be floating point")

        # (B, T, 3H) -> (3, B, heads, T, head_dim)
        qkv = self.qkv(hidden_states).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        # 반정밀도에서도 점수와 softmax는 float32로 계산한다.
        score_dtype = torch.float32 if query.dtype in (
            torch.float16, torch.bfloat16
        ) else query.dtype
        scores = (query.to(score_dtype) @ key.to(score_dtype).transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        future = torch.ones(length, length, dtype=torch.bool,
                            device=hidden_states.device).triu(diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
        weights = torch.softmax(scores, dim=-1).to(value.dtype)
        context = weights @ value
        # 여러 head의 결과를 원래 hidden 크기로 다시 합친다.
        context = context.transpose(1, 2).contiguous().reshape(batch, length, hidden)
        return self.out_proj(context)
