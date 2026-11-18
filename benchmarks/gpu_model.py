"""Model-only training benchmark; synthetic tokens are not a quality evaluation."""

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from configs.model_config import ModelConfig
from model.gpt import GPTModel


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["naive", "flash_attention_2"], default="naive")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--batch-size", type=positive, default=1)
    parser.add_argument("--seq-len", type=positive, default=128)
    parser.add_argument("--hidden-size", type=positive, default=256)
    parser.add_argument("--layers", type=positive, default=2)
    parser.add_argument("--heads", type=positive, default=4)
    parser.add_argument("--vocab-size", type=positive, default=1024)
    parser.add_argument("--warmup", type=positive, default=3)
    parser.add_argument("--steps", type=positive, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-revision", required=True, help="git rev-parse HEAD")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cpu" and (args.precision != "fp32" or args.backend != "naive"):
        parser.error("CPU smoke test requires naive + fp32")
    if args.backend == "flash_attention_2" and args.precision == "fp32":
        parser.error("use bf16/fp16 for both backends for a fair comparison")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; no GPU results were produced")
    if args.device == "cuda" and args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        parser.error("GPU does not support bf16; try fp16")
    if args.output.exists():
        parser.error("output already exists; choose a new path to preserve results")

    device = torch.device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
    config = ModelConfig(vocab_size=args.vocab_size, hidden_size=args.hidden_size,
                         num_layers=args.layers, num_heads=args.heads,
                         max_seq_len=args.seq_len, intermediate_size=4 * args.hidden_size,
                         attention_backend=args.backend)
    model = GPTModel(config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler(device.type, enabled=args.precision == "fp16")
    tokens = torch.randint(args.vocab_size, (args.batch_size, args.seq_len + 1), device=device)
    inputs, targets = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def step(check=False):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=dtype, enabled=args.precision != "fp32"):
            logits = model(inputs)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if check:
            if not torch.isfinite(loss).item():
                raise RuntimeError("nonfinite loss")
            for name, parameter in model.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
                    raise RuntimeError(f"missing/nonfinite gradient: {name}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        return loss.detach()

    # Validation is deliberately outside the timing window (it synchronizes).
    first_loss = float(step(check=True))
    for _ in range(args.warmup):
        step()
    optimizer.zero_grad(set_to_none=True)
    sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    scale_before = scaler.get_scale()
    start = time.perf_counter()
    for _ in range(args.steps):
        last_loss = step()
    sync()
    elapsed = time.perf_counter() - start
    memory = ({"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
              if device.type == "cuda" else None)
    scale_after = scaler.get_scale()
    final_loss = float(step(check=True))
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all().item():
            raise RuntimeError(f"nonfinite parameter: {name}")
    if scale_after < scale_before:
        raise RuntimeError("FP16 scaler decreased during timing; rerun after investigating overflow")
    result = {
        "status": "passed", "scope": "synthetic model-only forward/backward/AdamW",
        "not_language_quality_evaluation": True,
        "source_revision": args.source_revision, "config": asdict(config),
        "precision": args.precision, "seed": args.seed, "batch_size": args.batch_size,
        "warmup_steps": args.warmup, "measured_steps": args.steps,
        "seconds": elapsed, "tokens_per_second": args.batch_size * args.seq_len * args.steps / elapsed,
        "milliseconds_per_step": 1000 * elapsed / args.steps, "memory": memory,
        "first_loss": first_loss, "last_timed_loss": float(last_loss), "final_check_loss": final_loss,
        "finite_checks": "before and after timing; not every timed step",
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "python": platform.python_version(), "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
