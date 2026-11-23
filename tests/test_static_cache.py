import unittest
from unittest.mock import patch
import torch
from configs.model_config import ModelConfig
from model.gpt import GPTModel


class StaticCacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = GPTModel(ModelConfig(vocab_size=64, hidden_size=32,
            num_layers=2, num_heads=4, intermediate_size=64, max_seq_len=12)).eval()
        self.ids = torch.randint(0,64,(2,12))

    def test_equivalence_and_no_reallocation(self):
        cache = self.model.create_static_cache(12)
        with patch('torch.cat', side_effect=AssertionError('static cache must not concatenate')):
            logits, returned = self.model.forward_cached(self.ids[:,:3],cache,last_logits_only=True)
            self.assertIs(returned,cache)
            pointers = [(l.key.data_ptr(), l.value.data_ptr()) for l in cache.layers]
            torch.testing.assert_close(logits,self.model(self.ids[:,:3])[:,-1:],atol=1e-6,rtol=1e-5)
            for pos in range(3,12):
                logits,_=self.model.forward_cached(self.ids[:,pos:pos+1],cache,last_logits_only=True)
                torch.testing.assert_close(logits,self.model(self.ids[:,:pos+1])[:,-1:],atol=1e-6,rtol=1e-5)
                self.assertEqual(pointers,[(l.key.data_ptr(),l.value.data_ptr()) for l in cache.layers])
        self.assertEqual(cache.allocated_bytes,2*2*2*12*32*4)
        self.assertEqual(cache.length,12)
        with self.assertRaises(ValueError):self.model.forward_cached(self.ids[:,:1],cache)
        self.assertEqual(cache.length,12)
        cache.reset()
        logits,_=self.model.forward_cached(self.ids[:,2:6],cache)
        torch.testing.assert_close(logits,self.model(self.ids[:,2:6]))
        self.assertEqual(pointers,[(l.key.data_ptr(),l.value.data_ptr()) for l in cache.layers])

    def test_failure_resets_partial_cache(self):
        cache=self.model.create_static_cache()
        self.model.forward_cached(self.ids[:,:3],cache)
        with patch.object(self.model.blocks[1],'forward_cached',side_effect=RuntimeError('failure')):
            with self.assertRaises(RuntimeError):self.model.forward_cached(self.ids[:,3:4],cache)
        self.assertEqual(cache.length,0)
        logits,_=self.model.forward_cached(self.ids[:,:3],cache)
        torch.testing.assert_close(logits,self.model(self.ids[:,:3]))

    def test_validation_and_last_logits(self):
        for capacity in (0,-1,13,True,2.5):
            with self.assertRaises(ValueError):self.model.create_static_cache(capacity)
        cache=self.model.create_static_cache()
        self.model.train()
        with self.assertRaises(RuntimeError):self.model.forward_cached(self.ids[:,:3],cache)
        self.model.eval()
        other=GPTModel(self.model.config).eval()
        with self.assertRaises(ValueError):other.forward_cached(self.ids[:,:3],cache)
        logits,_=self.model.forward_cached(self.ids[:,:3],last_logits_only=True)
        torch.testing.assert_close(logits,self.model(self.ids[:,:3])[:,-1:])

    def test_precision_change_rejected(self):
        cache=self.model.create_static_cache()
        with torch.autocast('cpu',dtype=torch.bfloat16):
            self.model.forward_cached(self.ids[:,:3],cache)
            self.assertEqual(cache.layers[0].key.dtype,torch.bfloat16)
        with self.assertRaises(ValueError):self.model.forward_cached(self.ids[:,3:4],cache)
        self.assertEqual(cache.length,0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_flash_static(self):
        self.model.cuda();ids=self.ids.cuda()
        for b in self.model.blocks:b.attention.attention_backend='flash_attention_2'
        with torch.autocast('cuda',dtype=torch.float16):
            cache=self.model.create_static_cache()
            self.model.forward_cached(ids[:,:3],cache)
            for pos in range(3,12):
                logits,_=self.model.forward_cached(ids[:,pos:pos+1],cache)
                torch.testing.assert_close(logits,self.model(ids[:,:pos+1])[:,-1:],atol=2e-3,rtol=2e-2)
