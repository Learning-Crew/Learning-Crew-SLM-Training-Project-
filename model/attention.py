"""Multi-head causal self-attention for the GPT baseline."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel #Pytorch 내장 Flash Attention 모듈

from configs.model_config import ModelConfig
from model.kv_cache import StaticLayerCache


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
        self.attention_backend = getattr(config, "attention_backend", "naive")  #일단 기본값은 naive 입니다! -> 이전에 config랑 호환되도록 추가했어요!
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
        if self.attention_backend == "flash_attention_2":           #Flash Attention으로 해보려면 flash_attention_2로 설정하면 됩니다!
            context = self._flash_attention(query, key, value)
        else:
            context = self._naive_attention(query, key, value, length) #Flash Attention이 아니라 기본 Attention으로 하면 이걸로 QKV 계산합니다!
        # 여러 head의 결과를 원래 hidden 크기로 다시 합치게 됩니다!
        context = context.transpose(1, 2).contiguous().reshape(batch, length, hidden)
        return self.out_proj(context)

    def _flash_attention(self, query, key, value, *, is_causal=True):
        #PyTorch SDPA 안의 FlashAttention-2 커널 사용
        if not query.is_cuda:
            raise RuntimeError("flash_attention_2는 CUDA GPU에서만 사용할 수 있어요!")
        #FA2는 fp16, bf16만 받기 때문에 fp32 입력은 bf16으로 계산 후 원래 dtype으로 되돌립니다!
        #그래서 11/18에 배운 Mixed Precision처럼 bf16 or fp16으로 forward 후 backward는 fp32로 계산되지는 않습니다! -> 그 Attention 이외의 계산은 Mixed Precision을 따릅니다!
        input_dtype = query.dtype
        if input_dtype not in (torch.float16, torch.bfloat16):
            query, key, value = (x.to(torch.bfloat16) for x in (query, key, value))
        #FLASH_ATTENTION만 고정 해두었어요! (SDPA안에서 다른 거 불러오지 않도록 하길래 저도 비슷하게 구현했습니다!)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            context = F.scaled_dot_product_attention(query, key, value, is_causal=is_causal)
        return context.to(input_dtype)

    def _naive_attention(self, query, key, value, length):
        # 반정밀도에서도 점수와 softmax는 float32로 계산한다.
        score_dtype = torch.float32 if query.dtype in (
            torch.float16, torch.bfloat16
        ) else query.dtype
        scores = (query.to(score_dtype) @ key.to(score_dtype).transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        future = torch.ones(length, length, dtype=torch.bool,
                            device=query.device).triu(diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
        weights = torch.softmax(scores, dim=-1).to(value.dtype)
        return weights @ value

    @torch.no_grad()
    def forward_cached(self, hidden_states, past_key_value=None):
        """Inference-only prefill, then one new token per call.

        Cache is (key, value), each shaped (B, heads, past_length, head_dim).
        It belongs to one unchanged model, batch order and precision context.
        """
        if self.training:
            raise RuntimeError("KV cache requires eval()")
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states must have shape (B, T, H)")
        batch, length, hidden = hidden_states.shape
        if length < 1 or not hidden_states.is_floating_point():
            raise ValueError("expected a nonempty floating-point sequence")
        qkv = self.qkv(hidden_states).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        static_cache = isinstance(past_key_value, StaticLayerCache)
        has_past = past_key_value is not None and (
            not static_cache or past_key_value.length > 0
        )
        if static_cache:
            if has_past and length != 1:
                raise ValueError("cached decoding accepts exactly one new token")
            key, value = past_key_value.append(key, value)
        elif past_key_value is not None:
            if length != 1:
                raise ValueError("cached decoding accepts exactly one new token")
            if not isinstance(past_key_value, (tuple, list)) or len(past_key_value) != 2:
                raise ValueError("cache must contain key and value")
            past_key, past_value = past_key_value
            for cached in (past_key, past_value):
                if not isinstance(cached, torch.Tensor) or cached.ndim != 4:
                    raise ValueError("cache tensors must have shape (B, heads, T, head_dim)")
                if (cached.shape[0], cached.shape[1], cached.shape[3]) != (
                    batch, self.num_heads, self.head_dim
                ) or cached.shape[2] < 1:
                    raise ValueError("cache batch/head dimensions or length are invalid")
                if cached.device != key.device or cached.dtype != key.dtype:
                    raise ValueError("cache device and dtype must match current projections")
            if past_key.shape != past_value.shape:
                raise ValueError("cached key and value shapes must match")
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
        # A single newest query may attend to ALL cached keys. A rectangular
        # is_causal=True mask would incorrectly align that query to position 0.
        if self.attention_backend == "flash_attention_2":
            context = self._flash_attention(
                query, key, value, is_causal=not has_past
            )
        elif not has_past:
            context = self._naive_attention(query, key, value, length)
        else:
            score_dtype = torch.float32 if query.dtype in (
                torch.float16, torch.bfloat16
            ) else query.dtype
            scores = query.to(score_dtype) @ key.to(score_dtype).transpose(-2, -1)
            weights = torch.softmax(scores / math.sqrt(self.head_dim), dim=-1)
            context = weights.to(value.dtype) @ value
        context = context.transpose(1, 2).contiguous().reshape(batch, length, hidden)
        return self.out_proj(context), (key, value)
