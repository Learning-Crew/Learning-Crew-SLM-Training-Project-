import unittest

import torch
from torch import nn

from configs.model_config import ModelConfig
from model.embeddings import GPTEmbedding
from model.transformer_block import FeedForward, TransformerBlock


class TransformerBlockTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = ModelConfig(100, 32, 3, 4)

    def test_stacked_blocks_shape_and_gradient(self):
        embedding = GPTEmbedding(self.config, max_seq_len=8)
        stack = nn.Sequential(*(TransformerBlock(self.config) for _ in range(3)))
        output = stack(embedding(torch.tensor([[1, 2, 3], [3, 2, 1]])))
        self.assertEqual(tuple(output.shape), (2, 3, 32))
        output.square().mean().backward()
        for name, parameter in list(stack.named_parameters()) + list(embedding.named_parameters()):
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        for block in stack:
            self.assertGreater(block.attention.qkv.weight.grad.abs().sum().item(), 0)
            self.assertGreater(block.ffn.fc.weight.grad.abs().sum().item(), 0)

    def test_stacked_blocks_remain_causal(self):
        stack = nn.Sequential(*(TransformerBlock(self.config) for _ in range(3)))
        x = torch.randn(2, 6, 32)
        changed = x.clone()
        changed[:, 3:] = torch.randn_like(changed[:, 3:])
        torch.testing.assert_close(stack(x)[:, :3], stack(changed)[:, :3], rtol=0, atol=0)
        x.requires_grad_()
        stack(x)[:, :3].square().sum().backward()
        self.assertEqual(torch.count_nonzero(x.grad[:, 3:]).item(), 0)

    def test_residual_path_preserves_input_and_gradient(self):
        block = TransformerBlock(self.config)
        with torch.no_grad():
            for layer in (block.attention.out_proj, block.ffn.proj):
                layer.weight.zero_()
                layer.bias.zero_()
        x = torch.randn(2, 4, 32, requires_grad=True)
        output = block(x)
        torch.testing.assert_close(output, x, rtol=0, atol=0)
        output.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))

    def test_ffn_does_not_mix_positions(self):
        ffn = FeedForward(self.config)
        x = torch.randn(2, 4, 32)
        changed = x.clone()
        changed[:, 1] += 10
        torch.testing.assert_close(ffn(x)[:, [0, 2, 3]], ffn(changed)[:, [0, 2, 3]])

    def test_invalid_input(self):
        block = TransformerBlock(self.config)
        for shape in ((2, 32), (2, 3, 16), (2, 0, 32)):
            with self.assertRaises(ValueError):
                block(torch.zeros(shape))
        with self.assertRaises(TypeError):
            block(torch.zeros(2, 3, 32, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
