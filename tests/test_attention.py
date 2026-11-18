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

    def test_invalid_attention_backend(self):
        with self.assertRaises(ValueError):
            ModelConfig(100, 32, 2, 4, attention_backend="flash")

    def test_flash_attention_requires_cuda(self):
        #CPU에서는 FA2를 쓸 수 없으니 바로 에러가 나야 해요!
        model = CausalSelfAttention(ModelConfig(100, 32, 2, 4, attention_backend="flash_attention_2"))
        with self.assertRaises(RuntimeError):
            model(torch.randn(2, 5, 32))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU가 있어야 FA2 테스트 가능")
class FlashAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        #head_dim 64로 실제 학습 설정과 같은 크기에서 비교
        self.naive = CausalSelfAttention(ModelConfig(100, 256, 2, 4)).cuda()
        self.flash = CausalSelfAttention(
            ModelConfig(100, 256, 2, 4, attention_backend="flash_attention_2")).cuda()
        self.flash.load_state_dict(self.naive.state_dict())  #같은 가중치에서 계산 방식만 다르게

    def test_matches_naive_output_and_gradient(self):
        x = torch.randn(2, 128, 256, device="cuda")
        x_naive = x.clone().requires_grad_()
        x_flash = x.clone().requires_grad_()
        out_naive = self.naive(x_naive)
        out_flash = self.flash(x_flash)
        #FA2는 bf16으로 계산하니까 오차를 조금 허용
        torch.testing.assert_close(out_flash, out_naive, atol=2e-2, rtol=2e-2)
        out_naive.square().mean().backward()
        out_flash.square().mean().backward()
        torch.testing.assert_close(x_flash.grad, x_naive.grad, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(
            self.flash.qkv.weight.grad, self.naive.qkv.weight.grad, atol=2e-2, rtol=2e-2)

    def test_future_tokens_cannot_change_prefix(self):
        x = torch.randn(1, 16, 256, device="cuda")
        changed = x.clone()
        changed[:, 10:] = torch.randn(1, 6, 256, device="cuda")  #뒤쪽 토큰만 변경
        torch.testing.assert_close(self.flash(x)[:, :10], self.flash(changed)[:, :10], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
