"""Sequential GPU backend comparison. Synthetic throughput, not quality tuning."""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

def commands(revision, output):
    common = ["--precision", "bf16", "--source-revision", revision]
    yield "parity", [sys.executable, "-m", "benchmarks.compare_model", *common,
                     "--output", str(output / "parity.json")]
    for repeat in range(3):
        # Alternate order to reduce systematic warm/cold GPU bias.
        backends = ["naive", "flash_attention_2"]
        if repeat % 2:
            backends.reverse()
        for backend in backends:
            name = f"{backend}-{repeat}"
            yield name, [sys.executable, "-m", "benchmarks.gpu_model", *common,
                "--backend", backend, "--batch-size", "8", "--seq-len", "1024",
                "--hidden-size", "768", "--layers", "12", "--heads", "12",
                "--vocab-size", "32768", "--warmup", "5", "--steps", "20",
                "--output", str(output / f"{name}.json")]

def summarize(output):
    rows = []
    for backend in ["naive", "flash_attention_2"]:
        runs = [json.loads((output / f"{backend}-{i}.json").read_text()) for i in range(3)]
        if any(r["status"] != "passed" for r in runs):
            raise RuntimeError("Incomplete or failed measurements")
        rates = [r["tokens_per_second"] for r in runs]
        rows.append({"backend": backend, "median_tokens_per_second": statistics.median(rates),
                     "min_tokens_per_second": min(rates), "max_tokens_per_second": max(rates),
                     "max_peak_allocated_bytes": max(r["memory"]["peak_allocated_bytes"] for r in runs)})
    return {"scope": "model-only synthetic backend comparison",
            "not_final_training_config": True,
            "note": "No dataset, gradient accumulation or checkpointing. Benchmark uses its own optimizer defaults.",
            "measurements": rows,
            "flash_speed_ratio": rows[1]["median_tokens_per_second"] / rows[0]["median_tokens_per_second"]}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    jobs = list(commands(revision, args.output.resolve()))
    if args.dry_run:
        for name, command in jobs:
            print(name, json.dumps(command))
        return
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        parser.error("Use a clean checkout so the recorded commit identifies the code")
    import torch
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable; no tuning was run")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(json.dumps({
        "revision": revision, "jobs": jobs, "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "cuda": torch.version.cuda}, indent=2))
    for name, command in jobs:
        print("Running", name, flush=True)
        with (args.output / f"{name}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f"{name} failed; see {args.output / (name + '.log')}. No winner selected.")
    report = summarize(args.output)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
