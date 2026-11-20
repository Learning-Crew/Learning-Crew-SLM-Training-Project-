
from configs.model_config import ModelConfig
from configs.tokenizer_config import PretrainDatasetConfig, TokenizerConfig

from tokenizer.bpe_tokenizer import BPETokenizer

from model.gpt import GPTModel

from pathlib import Path

from data.tokenized_dataset import (
    TokenizedDataset,
    build_tokenized_file,
)

from data.load_preprocessed_data import load_preprocessed_data



def build_experiment(
    model_config: ModelConfig,
    dataset_config: PretrainDatasetConfig,
    tokenizer_config: TokenizerConfig,
):
    #설정값을 사용해서 GPT 모델과 BPE 토크나이저 생성
    model = GPTModel(model_config)
    tokenizer = BPETokenizer(
        config=tokenizer_config,
        dataset_config=dataset_config,
    )

    # 전처리된 FineWeb-Edu 데이터 경로
    preprocessed_dir = Path(
        dataset_config.preprocessed_dir
    ).expanduser()

    # 토큰화 결과를 저장할 디렉터리
    tokenized_dir = preprocessed_dir / "tokenized"
    tokenized_dir.mkdir(parents=True, exist_ok=True)

    train_token_file = tokenized_dir / "train.bin"
    validation_token_file = tokenized_dir / "valid.bin"

    # Train 데이터 토큰화 및 바이너리 저장
    if not train_token_file.exists():
        train_token_count = build_tokenized_file(
            documents=load_preprocessed_data("train", dataset_config),
            tokenizer=tokenizer,
            output_path=train_token_file,
        )
    else:
        train_token_count = train_token_file.stat().st_size // 2

    # Validation 데이터 토큰화 및 바이너리 저장
    if not validation_token_file.exists():
        validation_token_count = build_tokenized_file(
            documents=load_preprocessed_data("valid", dataset_config),
            tokenizer=tokenizer,
            output_path=validation_token_file,
        )
    else:
        validation_token_count = validation_token_file.stat().st_size // 2

    # 디스크 기반 Dataset 생성
    train_data = TokenizedDataset(
        token_file=train_token_file,
        token_count=train_token_count,
        seq_len=model_config.max_seq_len,
    )

    validation_data = TokenizedDataset(
        token_file=validation_token_file,
        token_count=validation_token_count,
        seq_len=model_config.max_seq_len,
    )

    # 데이터 규모 확인
    print(f"Train tokens: {train_token_count:,}")
    print(f"Validation tokens: {validation_token_count:,}")
    print(f"Train samples: {len(train_data):,}")
    print(f"Validation samples: {len(validation_data):,}")

    #Trainer에서 사용할 모델, 데이터, 데이터 식별자 반환
    return model, train_data, validation_data, "fineweb-edu-preprocessed-v1"
