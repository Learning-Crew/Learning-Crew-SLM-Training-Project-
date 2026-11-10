"""Pre-normalized GPT-style decoder block."""

import math

import torch
from torch import nn

from configs.model_config import ModelConfig
from model.attention import CausalSelfAttention


class FeedForward(nn.Module):
    """Apply H -> 4H -> H independently at each token position."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc = nn.Linear(config.hidden_size, 4 * config.hidden_size)
        self.activation = nn.GELU(approximate="tanh")
        self.proj = nn.Linear(4 * config.hidden_size, config.hidden_size)
        for layer in (self.fc, self.proj):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            nn.init.zeros_(layer.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(self.activation(self.fc(hidden_states)))


class TransformerBlock(nn.Module):
    """Preserve (B, T, H) while adding attention and token-wise processing.

    Pre-LN: x = x + Attention(LayerNorm(x)), then
    x = x + FFN(LayerNorm(x)). No padding mask or dropout in this baseline.
    config.num_layers must describe the full stack for residual scaling.
    The full GPT model will apply a final LayerNorm after all blocks.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.attention = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.ffn = FeedForward(config)
        # 블록을 여러 개 쌓을 때 잔차 가지의 초기 출력 크기를 줄인다.
        residual_std = 0.02 / math.sqrt(2 * config.num_layers)
        nn.init.normal_(self.attention.out_proj.weight, std=residual_std)
        nn.init.normal_(self.ffn.proj.weight, std=residual_std)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape (B, T, H)")
        if hidden_states.shape[-1] != self.hidden_size or hidden_states.shape[1] == 0:
            raise ValueError("hidden size must match config and T must be positive")
        if not hidden_states.is_floating_point():
            raise TypeError("hidden_states must be floating point")
        # 원래 입력을 더해 정보와 gradient가 흐르는 경로를 유지한다.
        hidden_states = hidden_states + self.attention(self.ln_1(hidden_states))
        return hidden_states + self.ffn(self.ln_2(hidden_states))
