import unittest

import torch
from torch.nn import functional as F

from configs.model_config import ModelConfig
from model.attention import CausalSelfAttention
from model.embeddings import GPTEmbedding


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = ModelConfig(100, 32, 2, 4)
        self.model = CausalSelfAttention(self.config)

    def test_future_tokens_cannot_change_prefix(self):
        embedding = GPTEmbedding(self.config, max_seq_len=8)
        first = self.model(embedding(torch.tensor([[1, 2, 3, 4, 5]])))
        changed = self.model(embedding(torch.tensor([[1, 2, 3, 8, 9]])))
        torch.testing.assert_close(first[:, :3], changed[:, :3], rtol=0, atol=0)
        self.assertFalse(torch.equal(first[:, 3:], changed[:, 3:]))

    def test_matches_pytorch_attention_reference(self):
        x = torch.randn(2, 5, 32)
        q, k, v = self.model.qkv(x).reshape(2, 5, 3, 4, 8).permute(
            2, 0, 3, 1, 4).unbind(0)
        reference = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        reference = self.model.out_proj(reference.transpose(1, 2).reshape(2, 5, 32))
        torch.testing.assert_close(self.model(x), reference, atol=1e-6, rtol=1e-5)

    def test_gradient_and_embedding_integration(self):
        embedding = GPTEmbedding(self.config, max_seq_len=8)
        result = self.model(embedding(torch.tensor([[1, 2, 3], [3, 2, 1]])))
        self.assertEqual(tuple(result.shape), (2, 3, 32))
        result.square().mean().backward()
        for parameter in list(self.model.parameters()) + list(embedding.parameters()):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(self.model.qkv.weight.grad.abs().sum().item(), 0)
        self.assertGreater(embedding.token_embedding.weight.grad.abs().sum().item(), 0)

    def test_no_future_gradient(self):
        x = torch.randn(1, 5, 32, requires_grad=True)
        self.model(x)[:, :3].square().sum().backward()
        self.assertEqual(torch.count_nonzero(x.grad[:, 3:]).item(), 0)
        self.assertGreater(x.grad[:, :3].abs().sum().item(), 0)

    def test_single_token(self):
        x = torch.randn(2, 1, 32)
        expected = self.model.out_proj(self.model.qkv(x).chunk(3, -1)[2])
        torch.testing.assert_close(self.model(x), expected)

    def test_invalid_input(self):
        for shape in ((2, 32), (2, 3, 16), (2, 0, 32)):
            with self.assertRaises(ValueError):
                self.model(torch.zeros(shape))
        with self.assertRaises(TypeError):
            self.model(torch.zeros(2, 3, 32, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
