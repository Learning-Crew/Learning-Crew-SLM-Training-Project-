"""GPT baseline: token IDs (B, T) -> unnormalized logits (B, T, V)."""

from dataclasses import asdict, replace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from configs.model_config import ModelConfig
from model.embeddings import GPTEmbedding
from model.transformer_block import TransformerBlock


class GPTModel(nn.Module):
    """Decoder-only model with learned positions and a tied output head.

    Loss and token shifting are handled by training/data, not forward().
    Dense, unpadded sequences only; optional inference via forward_cached().
    max_seq_len must be retained alongside ModelConfig in checkpoints.
    """

    def __init__(self, config: ModelConfig, *, max_seq_len: int | None = None):
        super().__init__()
        # Gradient Checkpointing 기본 비활성화
        self.gradient_checkpointing = False
        self.config = replace(config)
        self.max_seq_len = max_seq_len if max_seq_len is not None else getattr(config, "max_seq_len", 1024)
        if hasattr(self.config, "max_seq_len"):
            self.config.max_seq_len = self.max_seq_len
        self.embedding = GPTEmbedding(self.config, max_seq_len=self.max_seq_len)
        self.blocks = nn.ModuleList(
            TransformerBlock(self.config) for _ in range(self.config.num_layers)
        )
        self.final_norm = nn.LayerNorm(self.config.hidden_size, eps=getattr(self.config, "norm_eps", 1e-5))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embedding(input_ids)
        for block in self.blocks:
            #학습 중 Gradient Checkpointing이 활성화된 경우
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    block,
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        # 입력 embedding 가중치를 출력에도 공유한다 (weight tying).
        # softmax 없이 원점수 logits를 반환해야 CrossEntropyLoss에 연결된다.
        return F.linear(hidden_states, self.embedding.token_embedding.weight)

    def get_model_config(self) -> dict:
        """Return plain reconstruction metadata for the training checkpoint.

        Restore with GPTModel(ModelConfig(**meta['config']),
                              max_seq_len=meta['max_seq_len']).
        The training module owns checkpoint file I/O and optimizer state.
        """
        return {"config": asdict(self.config), "max_seq_len": self.max_seq_len}

    @torch.no_grad()
    def forward_cached(self, input_ids: torch.Tensor, past_key_values=None):
        """Return (logits, cache) for prefill or single-token decoding.

        Call eval() first. Use only newly appended IDs after prefill; reset
        cache to None for a different prompt. No padding or context rollover.
        The ordinary forward() and checkpoint parameter names are unchanged.
        """
        if self.training:
            raise RuntimeError("KV cache requires eval(); use forward() for training")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (B, T)")
        offset = 0
        if past_key_values is not None:
            if not isinstance(past_key_values, (tuple, list)) or len(past_key_values) != len(self.blocks):
                raise ValueError("cache must contain one key/value pair per layer")
            lengths = []
            for pair in past_key_values:
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ValueError("each layer cache must contain key and value")
                if any(not isinstance(t, torch.Tensor) or t.ndim != 4 for t in pair):
                    raise ValueError("cache tensors must be four-dimensional")
                if pair[0].shape != pair[1].shape or pair[0].shape[2] < 1:
                    raise ValueError("cache key/value shapes must match and be nonempty")
                lengths.append(pair[0].shape[2])
            if len(set(lengths)) != 1:
                raise ValueError("all layer caches must have the same sequence length")
            offset = lengths[0]
            if input_ids.shape[1] != 1:
                raise ValueError("pass exactly one new token after prefill")
        hidden_states = self.embedding(input_ids, position_offset=offset)
        present = []
        for index, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[index]
            hidden_states, layer_cache = block.forward_cached(hidden_states, past)
            present.append(layer_cache)
        logits = F.linear(self.final_norm(hidden_states), self.embedding.token_embedding.weight)
        return logits, tuple(present)
