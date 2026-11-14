import unittest

import torch

from configs.model_config import ModelConfig
from model.embeddings import GPTEmbedding


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = GPTEmbedding(ModelConfig(100, 32, 2, 4), max_seq_len=8)

    def test_shape_and_position_signal(self):
        output = self.model(torch.tensor([[7, 7, 7], [7, 7, 7]]))
        self.assertEqual(tuple(output.shape), (2, 3, 32))
        torch.testing.assert_close(output[0], output[1])
        self.assertFalse(torch.equal(output[0, 0], output[0, 1]))

    def test_gradient_reaches_both_tables(self):
        self.model(torch.tensor([[1, 2, 3]])).square().mean().backward()
        for table in (self.model.token_embedding, self.model.position_embedding):
            self.assertTrue(torch.isfinite(table.weight.grad).all())
            self.assertGreater(table.weight.grad.abs().sum().item(), 0)

    def test_sequence_boundary(self):
        self.assertEqual(self.model(torch.zeros(1, 8, dtype=torch.long)).shape[1], 8)
        for length in (0, 9):
            with self.assertRaises(ValueError):
                self.model(torch.zeros(1, length, dtype=torch.long))

    def test_invalid_inputs(self):
        with self.assertRaises(TypeError):
            self.model(torch.zeros(1, 3))
        with self.assertRaises(ValueError):
            self.model(torch.zeros(3, dtype=torch.long))
        for token in (-1, 100):
            with self.assertRaises(IndexError):
                self.model(torch.tensor([[token]]))


if __name__ == "__main__":
    unittest.main()
