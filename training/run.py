import json
import logging
import random
import numpy as np
from dataclasses import asdict, replace
from math import exp, isfinite
from pathlib import Path
from random import Random
from shutil import copyfile
from tqdm import tqdm

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.experimental.callback import Callback
from hydra.types import RunMode
from hydra.utils import get_method, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from configs.model_config import ModelConfig
from configs.training_config import TrainingConfig
from training.trainer import train
from configs.tokenizer_config import (
    PretrainDatasetConfig,
    TokenizerConfig,
)

logger = logging.getLogger(__name__)
SUBSET_FRACTION = 0.015
SUBSET_SEED = 42
# 교수님이 제공한 재현성 코드
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def select_subset(dataset: Dataset, split: str) -> Subset:
    #1.5%를 정수 연산으로 내림하고 데이터 전체를 읽지 않고 인덱스만 뽑기
    count = len(dataset) * 15 // 1000
    if count == 0:
        raise ValueError(f"{split} 데이터의 1.5%를 추출하면 샘플이 없습니다.")
    indices = sorted(Random(SUBSET_SEED).sample(range(len(dataset)), count))
    return Subset(dataset, indices)


def build_smoke_experiment(model_config: ModelConfig):
    #학습용과 검증용 토큰을 별도 시드로 만들어 같은 실험에서 반복 사용
    datasets = []
    for seed, count in [(0, 64), (1, 17)]:
        generator = torch.Generator().manual_seed(seed)
        starts = torch.randint(model_config.vocab_size, (count, 1), generator=generator)
        tokens = (starts + torch.arange(9)) % model_config.vocab_size
        datasets.append(TensorDataset(tokens[:, :-1], tokens[:, 1:]))
    model = nn.Sequential(
        nn.Embedding(model_config.vocab_size, model_config.hidden_size),
        nn.Linear(model_config.hidden_size, model_config.vocab_size),
    )
    return model, *datasets, "smoke-v1"


@torch.inference_mode()
def evaluate(model: nn.Module, dataset: Dataset, batch_size: int, device: str) -> float:
    model.eval()
    loss_sum = 0.0
    token_count = 0
    for inputs, targets in tqdm(
        DataLoader(dataset, batch_size=batch_size),
        desc="Validation",
        unit="batch"
    ):
        targets = targets.to(device)
        with torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=(
                torch.device(device).type == "cuda"
                and torch.cuda.is_bf16_supported()
            ),
        ):
            logits = model(inputs.to(device))
            loss = F.cross_entropy(
                logits.flatten(0, 1),
                targets.flatten(),
                reduction="sum",
            )

        loss_sum += loss.item()
        token_count += targets.numel()
    if token_count == 0:
        raise ValueError("검증할 토큰이 없습니다.")
    #마지막 배치가 작아도 토큰 수로 나눠 같은 비중으로 평가
    return loss_sum / token_count


class SweepSummary(Callback):
    def on_multirun_end(self, config: DictConfig, **kwargs) -> None:
        sweep_dir = Path(config.hydra.sweep.dir)
        jobs = sorted(
            (path for path in sweep_dir.glob("*") if path.is_dir() and path.name.isdecimal()),
            key=lambda path: int(path.name),
        )
        #실패한 후보가 있으면 완료된 일부 실험만으로 최적값을 고르지 않기 -> 나중에 최종 pretrain을 할 때 오류 나지 않도록!
        if not jobs or any(not (job / "metrics.json").is_file() for job in jobs):
            return
        results = []
        for job in jobs:
            metrics = json.loads((job / "metrics.json").read_text(encoding="utf-8"))
            results.append({"job": int(job.name), **metrics})
        results.sort(key=lambda result: result["validation_loss"])
        (sweep_dir / "summary.json").write_text(
            json.dumps({"best": results[0], "runs": results}, indent=2), encoding="utf-8"
        )
        logger.info("최적 조합은 %s에서 확인가능!", sweep_dir / "summary.json")


@hydra.main(
    config_path="../configs",
    config_name="experiment",
)
def main(config: DictConfig) -> None:
    hydra_config = HydraConfig.get()
    multirun = hydra_config.mode == RunMode.MULTIRUN
    if multirun and (
        config.resume_from is not None
        or OmegaConf.select(config, "init_from") is not None
    ):
        raise ValueError(
            "resume_from과 init_from은 단일 실행에서만 사용할 수 있습니다."
        )
    dataset_config = PretrainDatasetConfig(
        **OmegaConf.to_container(config.dataset, resolve=True)
    )

    tokenizer_config = TokenizerConfig(
        **OmegaConf.to_container(config.tokenizer, resolve=True)
    )
    model_config = ModelConfig(**OmegaConf.to_container(config.model, resolve=True))
    training_values = OmegaConf.to_container(config.training, resolve=True)
    configured_steps = training_values.pop("max_steps")
    if configured_steps is not None:
        training_values["max_steps"] = configured_steps
    training_config = TrainingConfig(**training_values)
    set_seed(training_config.seed)  #후보마다 같은 초기 가중치에서 시작하도록! 교수님 제공 코드로 변경

    smoke = config.factory is None
    if smoke:
        model, train_data, validation_data, dataset_id = build_smoke_experiment(model_config)
    else:
        factory = get_method(config.factory)
        model, train_data, validation_data, dataset_id = factory(
            model_config, 
            dataset_config=dataset_config,
            tokenizer_config=tokenizer_config,
            **OmegaConf.to_container(config.data, resolve=True)
        )

    use_subset = not smoke and (multirun or config.subset)
    original_counts = {"train": len(train_data), "validation": len(validation_data)}
    if use_subset:
        train_data = select_subset(train_data, "train")
        validation_data = select_subset(validation_data, "validation")
        dataset_id = f"{dataset_id}/subset-v1:{SUBSET_FRACTION}:seed={SUBSET_SEED}"
    if len(train_data) == 0 or len(validation_data) == 0:
        raise ValueError("train과 validation 데이터에 아무것도 없어요!")

    if use_subset or configured_steps is None:
        if smoke:
            steps = 20
        else:
            effective_batch_size = (
                training_config.batch_size
                * training_config.gradient_accumulation_steps
            )

            steps = (
                len(train_data) + effective_batch_size - 1
            ) // effective_batch_size

        training_config = replace(
            training_config,
            max_steps=steps,
        )

    output_dir = Path(hydra_config.runtime.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    resume_from = (
        Path(to_absolute_path(config.resume_from))
        if config.resume_from else None
    )

    init_from_value = OmegaConf.select(config, "init_from")
    init_from = (
        Path(to_absolute_path(init_from_value))
        if init_from_value is not None else None
    )

    epoch_offset = OmegaConf.select(
        config, "epoch_offset", default=0
    )
    previous_steps = OmegaConf.select(
        config, "previous_steps", default=0
    )

    completed_steps = train(
        model,
        train_data,
        model_config,
        training_config,
        device=config.device,
        dataset_id=dataset_id,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
        init_from=init_from,
        epoch_offset=epoch_offset,
        previous_steps=previous_steps,
    )

    final_checkpoint = checkpoint_dir / f"step_{completed_steps}.pt"
    #이미 완료된 체크포인트에서 재개하면 trainer가 새 파일을 쓰지 않음
    if resume_from is not None and not final_checkpoint.exists():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        copyfile(resume_from, final_checkpoint)

    validation_loss = evaluate(model, validation_data, training_config.batch_size, config.device)
    if not isfinite(validation_loss):
        raise ValueError("loss값이 유한한 값이 아닙니다! ")
        #해당 에러 계속 뜨면 Wandb같은거 사용해도 좋을 거 같네요!
    metrics = {
        "model": asdict(model_config),
        "training": asdict(training_config),
        "device": config.device,
        "dataset_id": dataset_id,
        "subset_fraction": SUBSET_FRACTION if use_subset else 1.0,
        "subset_seed": SUBSET_SEED if use_subset else None,
        "source_samples": original_counts,
        "samples": {"train": len(train_data), "validation": len(validation_data)},
        "completed_steps": completed_steps,
        "previous_steps": previous_steps,
        "total_completed_steps": previous_steps + completed_steps,
        "epoch_offset": epoch_offset,
        "validation_loss": validation_loss,
        "perplexity": exp(validation_loss),
        "checkpoint": str(final_checkpoint.relative_to(output_dir)),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
