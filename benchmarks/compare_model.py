"""Compare full GPT logits, loss and every parameter gradient; no speed claims."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch.nn import functional as F

from configs.model_config import ModelConfig
from model.gpt import GPTModel


def difference(actual, reference, tolerance):
    actual, reference = actual.detach().float(), reference.detach().float()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    delta = actual - reference
    absolute = float(delta.abs().max())
    relative = float(delta.norm() / reference.norm().clamp_min(1e-12))
    return {"finite": finite, "max_absolute_error": absolute,
            "relative_l2_error": relative, "reference_l2": float(reference.norm()),
            "passed": finite and (absolute <= 1e-7 or relative <= tolerance)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpu-self-check", action="store_true",
                   help="compare naive against itself; does NOT validate FlashAttention")
    p.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--relative-tolerance", type=float, default=0.05)
    p.add_argument("--source-revision", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not 0 < a.relative_tolerance < 1:
        p.error("relative tolerance must be between 0 and 1")
    if a.output.exists():
        p.error("output exists; choose a new path")
    device = "cpu" if a.cpu_self_check else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable; FlashAttention was not tested")
    if device == "cuda" and a.precision == "bf16" and not torch.cuda.is_bf16_supported():
        p.error("BF16 unsupported; try --precision fp16")
    torch.manual_seed(42)
    config = ModelConfig(vocab_size=256, hidden_size=128, num_layers=2,
                         num_heads=2, max_seq_len=32, intermediate_size=512)
    reference = GPTModel(config).to(device).eval()
    candidate = GPTModel(replace(config, attention_backend=(
        "naive" if a.cpu_self_check else "flash_attention_2"))).to(device).eval()
    candidate.load_state_dict(reference.state_dict())
    tokens = torch.randint(256, (2, 33), device=device)
    inputs, labels = tokens[:, :-1], tokens[:, 1:]
    dtype = torch.bfloat16 if a.precision == "bf16" else torch.float16

    def forward(model, ids):
        with torch.autocast(device, dtype=dtype, enabled=not a.cpu_self_check):
            return model(ids)

    outputs, losses = [], []
    for model in (reference, candidate):
        output = forward(model, inputs)
        loss = F.cross_entropy(output.float().flatten(0, 1), labels.flatten())
        loss.backward()
        outputs.append(output.detach())
        losses.append(loss.detach())
    metrics = {"logits": difference(outputs[1], outputs[0], a.relative_tolerance),
               "loss": difference(losses[1], losses[0], a.relative_tolerance)}
    ref_params = dict(reference.named_parameters())
    for name, parameter in candidate.named_parameters():
        ref_grad = ref_params[name].grad
        metrics["gradient:" + name] = (
            {"passed": False, "reason": "missing gradient"}
            if parameter.grad is None or ref_grad is None else
            difference(parameter.grad, ref_grad, a.relative_tolerance))
    changed = inputs.clone()
    changed[:, 16:] = (changed[:, 16:] + 1) % config.vocab_size
    with torch.no_grad():
        prefix = forward(candidate, changed)[:, :16]
    metrics["causal_prefix"] = difference(prefix, outputs[1][:, :16], 1e-5)
    passed = all(item["passed"] for item in metrics.values())
    result = {"status": "passed" if passed else "failed",
              "scope": "CPU naive self-check" if a.cpu_self_check else "GPU naive vs FlashAttention full GPT",
              "source_revision": a.source_revision, "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
              "precision": "fp32" if a.cpu_self_check else a.precision,
              "relative_tolerance": a.relative_tolerance,
              "absolute_floor": 1e-7, "metrics": metrics}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    # Nonfinite values remain failures; encode them as null for valid JSON.
    import math
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        return None if isinstance(value, float) and not math.isfinite(value) else value
    a.output.write_text(json.dumps(clean(result), indent=2, allow_nan=False) + "\n")
    print(result["scope"], result["status"], "->", a.output)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
