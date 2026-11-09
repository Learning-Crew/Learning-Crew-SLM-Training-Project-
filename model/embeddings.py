"""GPT-style token and learned absolute position embeddings."""

import torch
from torch import nn

from configs.model_config import ModelConfig


class GPTEmbedding(nn.Module):
    """Convert token IDs (B, T) to hidden vectors (B, T, H).

    max_seq_len is explicit because the shared ModelConfig does not yet
    define a context length. The final model must retain this value when
    saving its configuration. This module returns hidden states, not logits.
    """

    def __init__(self, config: ModelConfig, *, max_seq_len: int):
        super().__init__()
        if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int):
            raise TypeError("max_seq_len must be an integer")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.max_seq_len = max_seq_len
        # 토큰의 의미와 문장 안에서의 위치를 각각 학습한다.
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, config.hidden_size)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must use torch.long")
        sequence_length = input_ids.shape[1]
        if not 1 <= sequence_length <= self.max_seq_len:
            raise ValueError("sequence length must be between 1 and max_seq_len")
        # (T, H) 위치 벡터가 배치의 각 문장에 동일하게 더해진다.
        positions = torch.arange(sequence_length, device=input_ids.device)
        return self.token_embedding(input_ids) + self.position_embedding(positions)
