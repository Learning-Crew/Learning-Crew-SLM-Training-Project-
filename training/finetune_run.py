import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch

from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from torch.utils.data import DataLoader

from configs.model_config import ModelConfig
from configs.tokenizer_config import (
    TokenizerConfig,
    PretrainDatasetConfig,
)

from model.gpt import GPTModel
from tokenizer.bpe_tokenizer import BPETokenizer

from data.finetune_dataset import (
    load_finetune_data,
    group_split,
    FinetuneDataset,
    FinetuneCollator,
    get_answer_ids,
)

from training.finetune_trainer import train_finetune


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pretrained_model(path, device):
    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    model_config = ModelConfig(
        **state["model_config"]
    )

    model = GPTModel(model_config)
    model.load_state_dict(state["model"])

    print(
        "Pretrained steps:",
        state.get("total_completed_steps"),
    )

    del state

    return model.to(device)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="finetune",
)
def main(cfg: DictConfig):

    start = time.perf_counter()

    set_seed(cfg.training.seed)

    device = torch.device(cfg.device)

    output_dir = Path(
        HydraConfig.get().runtime.output_dir
    )

    print("Device:", device)
    print("Output directory:", output_dir)

    # 1. 사전학습 모델 로드
    print("\n===== LOAD PRETRAINED MODEL =====")

    model = load_pretrained_model(
        cfg.pretrained_checkpoint,
        device,
    )

    print(
        "Parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    # 2. 토크나이저 로드
    print("\n===== LOAD TOKENIZER =====")

    tokenizer = BPETokenizer(
        TokenizerConfig(
            vocab_size=cfg.tokenizer.vocab_size,
            vocab_path=cfg.tokenizer.vocab_path,
            merges_path=cfg.tokenizer.merges_path,
        ),
        PretrainDatasetConfig(),
    )

    assert tokenizer.vocab_size == model.config.vocab_size

    answer_ids = get_answer_ids(tokenizer)

    print("Answer IDs:", answer_ids)

    # 3. 데이터셋 로드 및 분리
    print("\n===== LOAD DATASET =====")

    dataset = load_finetune_data(
        cfg.dataset.path
    )

    train_ds, val_ds = group_split(
        dataset,
        val_ratio=cfg.dataset.val_ratio,
        seed=cfg.dataset.split_seed,
    )

    print("Train samples:", len(train_ds))
    print("Validation samples:", len(val_ds))

    # 4. Choice Likelihood 설정
    choice_weight = float(
        cfg.training.get("choice_weight", 0.0)
    )

    enable_choice_likelihood = choice_weight > 0

    print("Choice Likelihood weight:", choice_weight)

    # 5. 데이터 토큰화
    print("\n===== TOKENIZATION =====")

    train_data = FinetuneDataset(
        train_ds,
        tokenizer,
        max_seq_len=model.max_seq_len,
        shuffle_choices=cfg.dataset.shuffle_choices,
        enable_choice_likelihood=enable_choice_likelihood,
    )

    val_data = FinetuneDataset(
        val_ds,
        tokenizer,
        max_seq_len=model.max_seq_len,
        shuffle_choices=False,
        enable_choice_likelihood=False,
    )

    print("Shuffleable samples:", train_data.shuffleable_count)
    print("Train truncated:", train_data.truncated_count)
    print("Validation truncated:", val_data.truncated_count)

    if enable_choice_likelihood:
        print("Choice valid:", train_data.choice_valid_count)
        print("Choice skipped:", train_data.choice_skipped_count)

        print(
            "Choice coverage:",
            f"{train_data.choice_valid_count / len(train_data) * 100:.2f}%",
        )

        assert train_data.choice_valid_count > 0

    # 6. DataLoader 생성
    collator = FinetuneCollator(tokenizer.pad_id)

    generator = torch.Generator()
    generator.manual_seed(cfg.training.seed)

    train_loader = DataLoader(
        train_data,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
        num_workers=0,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
        num_workers=0,
        drop_last=False,
    )

    print(
        "\nPreparation elapsed:",
        round(time.perf_counter() - start, 2),
        "s",
    )

    # 7. 파인튜닝 실행
    train_finetune(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        answer_ids=answer_ids,
        config=cfg.training,
        device=device,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()