"""GPT baseline: token IDs (B, T) -> unnormalized logits (B, T, V)."""

from dataclasses import asdict, replace

import torch
from torch import nn
from torch.nn import functional as F

from configs.model_config import ModelConfig
from model.embeddings import GPTEmbedding
from model.transformer_block import TransformerBlock


class GPTModel(nn.Module):
    """Decoder-only model with learned positions and a tied output head.

    Loss and token shifting are handled by training/data, not forward().
    Dense, unpadded sequences only; no KV cache in this baseline.
    max_seq_len must be retained alongside ModelConfig in checkpoints.
    """

    def __init__(self, config: ModelConfig, *, max_seq_len: int):
        super().__init__()
        self.config = replace(config)
        self.max_seq_len = max_seq_len
        self.embedding = GPTEmbedding(self.config, max_seq_len=max_seq_len)
        self.blocks = nn.ModuleList(
            TransformerBlock(self.config) for _ in range(self.config.num_layers)
        )
        self.final_norm = nn.LayerNorm(self.config.hidden_size, eps=1e-5)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embedding(input_ids)
        for block in self.blocks:
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
