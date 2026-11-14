import io
import unittest

import torch
from torch.nn import functional as F

from configs.model_config import ModelConfig
from model.gpt import GPTModel


class GPTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = ModelConfig(16, 16, 2, 4)
        self.model = GPTModel(self.config, max_seq_len=8)

    def test_interface_and_all_parameter_gradients(self):
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        targets = torch.tensor([[2, 3, 4], [5, 6, 7]])
        logits = self.model(ids)
        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(tuple(logits.shape), (2, 3, 16))
        F.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1)).backward()
        for name, p in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())

    def test_logits_are_causal(self):
        a = self.model(torch.tensor([[1, 2, 3, 4, 5]]))
        b = self.model(torch.tensor([[1, 2, 3, 8, 9]]))
        torch.testing.assert_close(a[:, :3], b[:, :3], atol=0, rtol=0)
        self.assertFalse(torch.equal(a[:, 3:], b[:, 3:]))

    def test_checkpoint_roundtrip(self):
        ids = torch.tensor([[1, 2, 3]])
        checkpoint = {"model_config": self.model.get_model_config(),
                      "model_state_dict": self.model.state_dict()}
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=True)
        meta = loaded["model_config"]
        restored = GPTModel(ModelConfig(**meta["config"]), max_seq_len=meta["max_seq_len"])
        restored.load_state_dict(loaded["model_state_dict"])
        torch.testing.assert_close(self.model(ids), restored(ids), atol=0, rtol=0)

    def test_can_learn_tiny_next_token_task(self):
        # 검증용 작은 반복 데이터; 실제 학습 파이프라인을 대체하지 않는다.
        ids = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
        targets = torch.tensor([[2, 3, 4, 5], [2, 3, 4, 5]])
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        def loss():
            return F.cross_entropy(self.model(ids).reshape(-1, 16), targets.reshape(-1))
        initial = loss().item()
        for _ in range(40):
            optimizer.zero_grad(set_to_none=True)
            loss().backward()
            optimizer.step()
        final = loss().item()
        self.assertLess(final, initial * 0.25)
        print(f"\nTiny next-token loss: {initial:.4f} -> {final:.4f}")

    def test_length_boundaries_and_config_snapshot(self):
        for length in (1, 8):
            self.assertEqual(self.model(torch.ones(1, length, dtype=torch.long)).shape,
                             (1, length, 16))
        with self.assertRaises(ValueError):
            self.model(torch.ones(1, 9, dtype=torch.long))
        self.config.hidden_size = 999
        self.assertEqual(self.model.get_model_config()["config"]["hidden_size"], 16)


if __name__ == "__main__":
    unittest.main()
