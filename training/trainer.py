import logging
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset

from configs.model_config import ModelConfig
from configs.training_config import TrainingConfig
from training.checkpoint import load_checkpoint, save_checkpoint


logger = logging.getLogger(__name__)


def train(
    model: nn.Module,
    dataset: Dataset[tuple[Tensor, Tensor]],
    model_config: ModelConfig,
    training_config: TrainingConfig,
    *,
    device: str | torch.device,
    dataset_id: str,
    checkpoint_dir: str | Path,
    resume_from: str | Path | None = None,
) -> int:

    dataset_length = len(dataset)
    if dataset_length == 0:
        raise ValueError("학습할 데이터가 없습니다!")

    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    elif device.type == "cpu":
        device = torch.device("cpu")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("CPU, CUDA만 가능해요!")

    config = training_config
    #마지막에 배치 크기보다 적은 샘플이 남아도 한 배치로 계산
    batches_per_epoch = (dataset_length + config.batch_size - 1) // config.batch_size
    run_metadata = {
        "model_config": asdict(model_config),
        "training_config": asdict(config),
        "dataset_id": dataset_id,
        "dataset_length": dataset_length,
        "device": str(device),
    }
    state = None if resume_from is None else load_checkpoint(resume_from, run_metadata)
    global_step = 0
    epoch = 0
    next_batch = 0
    if state is not None:
        global_step = state["global_step"]
        #완료한 스텝을 epoch당 배치 수로 나눠 재개할 epoch, 배치 위치 확인
        epoch, next_batch = divmod(global_step, batches_per_epoch)
        if not 0 <= global_step <= config.max_steps or (epoch, next_batch) != (
            state["next_epoch"], state["next_batch"]
        ):
            raise ValueError("체크포인트에 다른 학습 위치 확인!")

    model.to(device)  #체크포인트는 CPU로 읽고 실제 학습은 지정한 장치에서 실행
    model.train()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.max_steps)
    if state is not None:
        #모델, 옵티마이저, 스케줄러 상태와 dropout 등에 쓰이는 난수 상태 복원
        model.load_state_dict(state["model"])
        scheduler.load_state_dict(state["scheduler"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng_state"], device)
        del state

    checkpoint_dir = Path(checkpoint_dir)
    #데이터 순서를 섞는 난수 생성기를 따로 사용 -> 같은 시드로 만드는 것이라 같은 순서가 나와요!
    generator = torch.Generator(device="cpu")
    while global_step < config.max_steps:
        #같은 epoch에서는 같은 순서로 섞고, 재개할 배치부터 읽기
        generator.manual_seed(config.seed + epoch)
        indices = torch.randperm(dataset_length, generator=generator, device="cpu")
        for batch_index in range(next_batch, batches_per_epoch):
            start = batch_index * config.batch_size
            batch_indices = indices[start : start + config.batch_size].tolist()
            inputs, targets = zip(*(dataset[index] for index in batch_indices), strict=True)
            inputs = torch.stack(inputs).to(device)
            targets = torch.stack(targets).to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            #배치와 시퀀스 차원을 합쳐 토큰마다 정답과 비교
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            loss.backward()
            learning_rate = optimizer.param_groups[0]["lr"]  #이번 업데이트에 사용한 학습률을 기록
            optimizer.step()
            scheduler.step()
            global_step += 1

            logger.info("step=%d loss=%.9g lr=%.9g", global_step, loss.item(), learning_rate)
            if global_step % config.checkpoint_every == 0 or global_step == config.max_steps:
                #다음에 읽을 배치 위치 저장. epoch 끝이면 다음 epoch의 첫 배치
                saved_epoch, saved_batch = divmod(global_step, batches_per_epoch)
                save_checkpoint(
                    checkpoint_dir / f"step_{global_step}.pt",
                    {
                        **run_metadata,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "global_step": global_step,
                        "next_epoch": saved_epoch,
                        "next_batch": saved_batch,
                        "cpu_rng_state": torch.get_rng_state(),
                        "cuda_rng_state": (
                            torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                        ),
                    },
                )
            if global_step == config.max_steps:
                break

        epoch += 1
        next_batch = 0

    return global_step
