"""Compare cached inference to full-prefix inference, without training data."""
import unittest
from unittest.mock import patch
import torch
from configs.model_config import ModelConfig
from model.gpt import GPTModel


class KVCacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = GPTModel(ModelConfig(vocab_size=32, hidden_size=16,
            num_layers=2, num_heads=4, intermediate_size=32, max_seq_len=8)).eval()
        self.ids = torch.randint(0, 32, (2, 8))

    def test_prefill_and_every_decode_position(self):
        for prefix in (1, 3, 7):
            logits, cache = self.model.forward_cached(self.ids[:, :prefix])
            torch.testing.assert_close(logits, self.model(self.ids[:, :prefix]))
            old_key = cache[0][0].clone()
            for pos in range(prefix, 8):
                # Project only the newly supplied token, not the full prefix.
                with patch.object(self.model.blocks[0].attention.qkv, 'forward',
                                  wraps=self.model.blocks[0].attention.qkv.forward) as qkv:
                    logits, cache = self.model.forward_cached(self.ids[:, pos:pos+1], cache)
                    self.assertEqual(qkv.call_args.args[0].shape[1], 1)
                torch.testing.assert_close(logits[:, -1], self.model(self.ids[:, :pos+1])[:, -1],
                                           atol=1e-6, rtol=1e-5)
                self.assertEqual(cache[0][0].shape, (2, 4, pos+1, 4))
                self.assertFalse(cache[0][0].requires_grad)
            torch.testing.assert_close(cache[0][0][:, :, :prefix], old_key)

    def test_cache_rejects_invalid_inputs(self):
        _, cache = self.model.forward_cached(self.ids[:, :3])
        invalid = [cache[:1], (cache[0], (cache[1][0][:,:,:2], cache[1][1][:,:,:2])),
                   ((cache[0][0].double(), cache[0][1].double()), cache[1]),
                   ((cache[0][0][:1], cache[0][1][:1]), cache[1])]
        for item in invalid:
            with self.assertRaises(ValueError):
                self.model.forward_cached(self.ids[:, 3:4], item)
        with self.assertRaises(ValueError):
            self.model.forward_cached(self.ids[:, 3:5], cache)
        _, full = self.model.forward_cached(self.ids)
        with self.assertRaises(ValueError):
            self.model.forward_cached(self.ids[:, :1], full)
        self.model.train()
        with self.assertRaises(RuntimeError):
            self.model.forward_cached(self.ids)

    def test_greedy_generation_matches_full_prefix(self):
        tokens = self.ids[:, :2]
        full = tokens.clone()
        logits, cache = self.model.forward_cached(tokens)
        for _ in range(5):
            next_token = logits[:, -1].argmax(-1, keepdim=True)
            tokens = torch.cat((tokens, next_token), dim=1)
            full = torch.cat((full, self.model(full)[:, -1].argmax(-1, keepdim=True)), dim=1)
            self.assertTrue(torch.equal(tokens, full))
            logits, cache = self.model.forward_cached(next_token, cache)

    def test_training_and_state_dict_are_unchanged(self):
        state = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.forward_cached(self.ids)
        self.assertEqual(set(state), set(self.model.state_dict()))
        for key, value in state.items():
            torch.testing.assert_close(value, self.model.state_dict()[key], atol=0, rtol=0)
        self.model.load_state_dict(state, strict=True)
        self.model.train()
        for checkpointing in (False, True):
            self.model.zero_grad(set_to_none=True)
            self.model.gradient_checkpointing = checkpointing
            self.model(self.ids).square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in self.model.parameters()))

    def test_flash_decode_mask_alignment(self):
        # Check the exact SDPA masking contract on CPU; not a CUDA speed test.
        for block in self.model.blocks:
            block.attention.attention_backend = 'flash_attention_2'
        def sdpa(query, key, value, *, is_causal=True):
            return torch.nn.functional.scaled_dot_product_attention(query, key, value,
                                                                    is_causal=is_causal)
        with patch('model.attention.CausalSelfAttention._flash_attention', side_effect=sdpa):
            logits, cache = self.model.forward_cached(self.ids[:, :3])
            for pos in range(3, 8):
                logits, cache = self.model.forward_cached(self.ids[:, pos:pos+1], cache)
                torch.testing.assert_close(logits[:, -1], self.model(self.ids[:, :pos+1])[:, -1],
                                           atol=1e-6, rtol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_flash_decode(self):
        self.model.cuda()
        ids = self.ids.cuda()
        for block in self.model.blocks:
            block.attention.attention_backend = 'flash_attention_2'
        with torch.autocast('cuda', dtype=torch.float16):
            _, cache = self.model.forward_cached(ids[:, :3])
            for pos in range(3, 8):
                logits, cache = self.model.forward_cached(ids[:, pos:pos+1], cache)
                torch.testing.assert_close(logits[:, -1], self.model(ids[:, :pos+1])[:, -1],
                                           atol=2e-3, rtol=2e-2)


if __name__ == '__main__':
    unittest.main()
