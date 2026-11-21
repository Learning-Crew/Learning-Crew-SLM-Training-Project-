
from configs.model_config import ModelConfig
from configs.tokenizer_config import PretrainDatasetConfig, TokenizerConfig

from tokenizer.bpe_tokenizer import BPETokenizer

from model.gpt import GPTModel

from pathlib import Path

from data.tokenized_dataset import TokenizedDataset

import hashlib
import json

def load_verified_tokens(split, tokenized_dir, tokenizer_config):
    token_file = tokenized_dir / f"{split}.bin"
    meta_file = tokenized_dir / f"{split}.meta.json"

    if not token_file.is_file() or not meta_file.is_file():
        raise FileNotFoundError(f"{split} 토큰화 파일이 없습니다.")

    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    def file_hash(path):
        h = hashlib.sha256()
        with open(Path(path).expanduser(), "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    expected = {
        "split": split,
        "vocab_size": tokenizer_config.vocab_size,
        "dtype": "uint16",
        "vocab_sha256": file_hash(tokenizer_config.vocab_path),
        "merges_sha256": file_hash(tokenizer_config.merges_path),
    }

    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"{split} 메타데이터 불일치: {key}")

    token_count = metadata.get("token_count")

    if type(token_count) is not int or token_count <= 0:
        raise ValueError(f"{split} 토큰 수가 잘못되었습니다.")

    if token_file.stat().st_size != token_count * 2:
        raise ValueError(f"{split} 파일 크기가 일치하지 않습니다.")

    return token_count

def build_experiment(
    model_config: ModelConfig,
    dataset_config: PretrainDatasetConfig,
    tokenizer_config: TokenizerConfig,
):
    #설정값을 사용해서 GPT 모델과 BPE 토크나이저 생성
    model = GPTModel(model_config)

    # 모델 파라미터 수 확인
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Total Parameters: {total_params:,}")
    print(f"[Model] Size: {total_params / 1e6:.2f}M")
    tokenizer = BPETokenizer(
        config=tokenizer_config,
        dataset_config=dataset_config,
    )
    if model_config.vocab_size != tokenizer_config.vocab_size:
        raise ValueError(
            "모델과 토크나이저의 vocab_size가 일치하지 않습니다."
        )

    # 전처리된 FineWeb-Edu 데이터 경로
    preprocessed_dir = Path(
        dataset_config.preprocessed_dir
    ).expanduser()

    # 토큰화 결과를 저장할 디렉터리
    tokenized_dir = (
        preprocessed_dir
        / "tokenized"
        / f"bpe_{tokenizer_config.vocab_size}"
    )
    tokenized_dir.mkdir(parents=True, exist_ok=True)

    train_token_file = tokenized_dir / "train.bin"
    validation_token_file = tokenized_dir / "valid.bin"

    # 기존 토큰화 파일 검증 및 로딩
    train_token_count = load_verified_tokens(
        "train", tokenized_dir, tokenizer_config
    )

    validation_token_count = load_verified_tokens(
        "valid", tokenized_dir, tokenizer_config
    )

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
