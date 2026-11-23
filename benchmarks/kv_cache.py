"""Reproducible full-prefix/dynamic/static inference comparison on synthetic IDs.

Run from repository root: python -m benchmarks.kv_cache --device cpu
Measures prefill + teacher-forced decode, NOT language quality or pretraining.
"""
import argparse
import contextlib
import json
import platform
import statistics
import time
import torch
from configs.model_config import ModelConfig
from model.gpt import GPTModel


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--dtype',choices=['float32','float16','bfloat16'],default='float32')
    p.add_argument('--backend',choices=['naive','flash_attention_2'],default='naive')
    p.add_argument('--hidden',type=int,default=128)
    p.add_argument('--layers',type=int,default=2)
    p.add_argument('--heads',type=int,default=4)
    p.add_argument('--vocab',type=int,default=1024)
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--prompt',type=int,default=64)
    p.add_argument('--decode',type=int,default=32)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--output',default=None)
    a=p.parse_args()
    if min(a.hidden,a.layers,a.heads,a.vocab,a.batch,a.prompt,a.decode,a.repeats)<1:
        p.error('numeric settings must be positive')
    if a.backend=='flash_attention_2' and a.device!='cuda':p.error('Flash requires CUDA')
    torch.manual_seed(42)
    if a.device=='cuda':torch.cuda.manual_seed_all(42)
    model=GPTModel(ModelConfig(vocab_size=a.vocab,hidden_size=a.hidden,
        num_layers=a.layers,num_heads=a.heads,intermediate_size=4*a.hidden,
        max_seq_len=a.prompt+a.decode,attention_backend=a.backend)).to(a.device).eval()
    ids=torch.randint(a.vocab,(a.batch,a.prompt+a.decode),device=a.device)
    modes=['full','dynamic','static']
    def sync():
        if a.device=='cuda':torch.cuda.synchronize()
    def autocast():
        return contextlib.nullcontext() if a.dtype=='float32' else torch.autocast(a.device,dtype=getattr(torch,a.dtype))
    @torch.no_grad()
    def run(mode):
        with autocast():
            if mode=='full':
                logits=model(ids[:,:a.prompt])[:,-1:]
                for end in range(a.prompt+1,ids.shape[1]+1):logits=model(ids[:,:end])[:,-1:]
            else:
                cache=model.create_static_cache(ids.shape[1]) if mode=='static' else None
                logits,cache=model.forward_cached(ids[:,:a.prompt],cache,last_logits_only=True)
                for pos in range(a.prompt,ids.shape[1]):
                    logits,cache=model.forward_cached(ids[:,pos:pos+1],cache,last_logits_only=True)
            return logits
    reference=run('full')
    for mode in modes:
        torch.testing.assert_close(run(mode),reference,atol=2e-3 if a.dtype!='float32' else 1e-5,
                                   rtol=2e-2 if a.dtype!='float32' else 1e-4)
    timings={m:[] for m in modes};memory={m:[] for m in modes}
    for repeat in range(a.repeats):
        for mode in modes[repeat%3:]+modes[:repeat%3]:
            sync()
            if a.device=='cuda':torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter();result=run(mode);sync()
            timings[mode].append(time.perf_counter()-start)
            if a.device=='cuda':memory[mode].append(torch.cuda.max_memory_allocated())
            del result
    report={'settings':vars(a),'torch':torch.__version__,'python':platform.python_version(),
            'gpu':torch.cuda.get_device_name() if a.device=='cuda' else None,
            'seed':42,'threads':torch.get_num_threads(),'measurement':'prefill + fixed-token decode',
            'results':{m:{'seconds':timings[m],'median_seconds':statistics.median(timings[m]),
                'peak_allocated_bytes':max(memory[m]) if memory[m] else None} for m in modes}}
    text=json.dumps(report,indent=2)
    print(text)
    if a.output:
        from pathlib import Path
        Path(a.output).write_text(text+'\n')

if __name__=='__main__':main()
