import unittest
import torch
from torch.utils.data import DataLoader
from configs.model_config import ModelConfig
from model.gpt import GPTModel
from evaluation.evaluator import evaluate

class IntegrationTests(unittest.TestCase):
    def test_config_and_evaluator(self):
        config = ModelConfig(vocab_size=16, hidden_size=16, num_layers=2,
                             num_heads=4, max_seq_len=8,
                             intermediate_size=48, norm_eps=1e-4)
        model = GPTModel(config)
        self.assertEqual(model.embedding.max_seq_len, 8)
        self.assertEqual(model.blocks[0].ffn.fc.out_features, 48)
        self.assertEqual(model.blocks[0].ln_1.eps, 1e-4)
        self.assertEqual(model.final_norm.eps, 1e-4)
        data = [(torch.tensor([1,2,3]), torch.tensor([2,3,4])),
                (torch.tensor([2,3,4]), torch.tensor([3,4,-100]))]
        one = evaluate(model, DataLoader(data, batch_size=1), torch.device('cpu'))
        two = evaluate(model, DataLoader(data, batch_size=2), torch.device('cpu'))
        for key in one:
            self.assertAlmostEqual(one[key], two[key], places=5)
        self.assertGreater(one['loss'], 0)
        override = GPTModel(config, max_seq_len=4)
        self.assertEqual(override.get_model_config()['config']['max_seq_len'], 4)
        self.assertEqual(config.max_seq_len, 8)
