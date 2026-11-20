import logging
from dataclasses import asdict
from pathlib import Path
from math import cos, pi

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
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

    # Gradient Checkpointing 설정을 모델에 전달
    if config.gradient_checkpointing:
        if not hasattr(model, "gradient_checkpointing"):
            raise ValueError(
                "현재 모델은 Gradient Checkpointing을 지원하지 않습니다."
            )

    if hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = config.gradient_checkpointing

    logger.info(
        "Gradient Checkpointing: %s",
        "enabled" if config.gradient_checkpointing else "disabled"
    )
    
    # Mixed Precision 설정
    # bf16은 지원하는 CUDA GPU에서만 활성화
    use_amp = (
        device.type == "cuda"
        and config.mixed_precision == "bf16"
    )

    if use_amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "현재 GPU는 bf16 Mixed Precision을 지원하지 않습니다."
        )

    logger.info(
        "Mixed Precision: %s",
        "bf16" if use_amp else "disabled"
    )

    # 한 번에 GPU에서 처리하는 미니배치 크기
    batch_size = config.batch_size

    # 몇 개의 미니배치마다 모델을 업데이트
    accumulation_steps = config.gradient_accumulation_steps

    #마지막에 배치 크기보다 적은 샘플이 남아도 한 배치로 계산
    # 한 epoch의 미니배치 수
    micro_batches_per_epoch = (
        dataset_length + batch_size - 1
    ) // batch_size

    # 한 epoch의 optimizer 업데이트 횟수
    batches_per_epoch = (
        micro_batches_per_epoch + accumulation_steps - 1
    ) // accumulation_steps

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
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.adam_betas,
        eps=config.adam_eps,
        )
    # Warmup + Cosine Learning Rate Scheduler 구현!
    def lr_lambda(step: int) -> float:
        warmup = min(config.warmup_steps, config.max_steps)

        # 1. Warmup: 학습률을 점진적으로 증가
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup

        # 2. Cosine Decay: 학습률을 점진적으로 감소
        decay_steps = max(1, config.max_steps - warmup)

        progress = min(
            1.0,
            max(0.0, (step - warmup) / decay_steps)
        )

        min_ratio = config.min_lr / config.learning_rate

        cosine = 0.5 * (1.0 + cos(pi * progress))

        return min_ratio + (1.0 - min_ratio) * cosine


    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lr_lambda
    )

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
        # 같은 epoch에서는 같은 순서로 섞고, 재개할 배치부터 읽기
        generator.manual_seed(config.seed + epoch)
        indices = torch.randperm(
            dataset_length,
            generator=generator,
            device="cpu"
        )

        for batch_index in range(next_batch, batches_per_epoch):

            # 이번 optimizer 업데이트에서 사용할 미니배치 범위 계산
            micro_start = batch_index * accumulation_steps
            micro_end = min(
                micro_start + accumulation_steps,
                micro_batches_per_epoch
            )

            # 마지막에 남은 샘플까지 고려하여 실제 샘플 수 계산
            update_start = micro_start * batch_size
            update_end = min(
                micro_end * batch_size,
                dataset_length
            )
            total_samples = update_end - update_start

            # Gradient Accumulation 시작 전에 기울기 초기화
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0

            # 여러 미니배치에 대해 Gradient 누적
            for micro_index in range(micro_start, micro_end):
                start = micro_index * batch_size
                end = min(start + batch_size, dataset_length)

                batch_indices = indices[start:end].tolist()

                inputs, targets = zip(
                    *(dataset[index] for index in batch_indices),
                    strict=True
                )

                inputs = torch.stack(inputs).to(device)
                targets = torch.stack(targets).to(device)

                # Mixed Precision 적용
                # CUDA bf16 지원 시 순전파와 Loss 계산에 autocast 사용
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_amp
                ):
                    # GPT 모델 순전파
                    logits = model(inputs)

                    # 배치와 시퀀스 차원을 합쳐 토큰마다 정답과 비교
                    loss = F.cross_entropy(
                        logits.flatten(0, 1),
                        targets.flatten()
                    )

                # 미니배치 크기에 따라 Loss를 조정하여 평균 Gradient 계산
                # 마지막 미니배치가 작아도 올바른 비율로 반영
                current_batch_size = len(batch_indices)
                scaled_loss = loss * (
                    current_batch_size / total_samples
                )

                # optimizer.step() 없이 역전파하여 Gradient 누적
                scaled_loss.backward()

                # 학습 로그 출력을 위한 Loss 합산
                total_loss += loss.item() * current_batch_size

            # Gradient 크기가 너무 커지지 않도록 제한
            # 모든 미니배치의 Gradient 누적이 끝난 뒤 한 번만 실행
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.max_grad_norm
            )

            # 이번 업데이트에 사용한 학습률을 기록
            learning_rate = optimizer.param_groups[0]["lr"]

            # Gradient Accumulation 완료 후 모델 가중치 업데이트
            optimizer.step()

            # 모델 업데이트 횟수에 맞춰 Scheduler 실행
            scheduler.step()

            # global_step은 미니배치 수가 아니라 optimizer 업데이트 횟수
            global_step += 1

            # 이번 업데이트에서 사용한 전체 샘플의 평균 Loss
            average_loss = total_loss / total_samples

            if global_step % config.log_every == 0 or global_step == config.max_steps:
                logger.info(
                    "step=%d loss=%.9g lr=%.9g",
                    global_step,
                    average_loss,
                    learning_rate
                )

            if global_step % config.checkpoint_every == 0 or global_step == config.max_steps:
                # 다음에 읽을 배치 위치 저장. epoch 끝이면 다음 epoch의 첫 배치
                # 이제 batch_index는 optimizer 업데이트 단위로 계산
                saved_epoch, saved_batch = divmod(
                    global_step,
                    batches_per_epoch
                )

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
                            torch.cuda.get_rng_state(device)
                            if device.type == "cuda"
                            else None
                        ),
                    },
                )

            if global_step == config.max_steps:
                break

        epoch += 1
        next_batch = 0

    return global_step