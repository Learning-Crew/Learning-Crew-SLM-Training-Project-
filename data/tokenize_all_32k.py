
from pathlib import Path
from datetime import datetime
import json
import hashlib

from configs.tokenizer_config import (
    PretrainDatasetConfig,
    TokenizerConfig,
)
from tokenizer.bpe_tokenizer import BPETokenizer
from data.load_preprocessed_data import load_preprocessed_data
from data.tokenized_dataset import build_tokenized_file


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    home = Path.home()

    dataset_config = PretrainDatasetConfig(
        preprocessed_dir=str(home / "preprocessed"),
    )

    tokenizer_config = TokenizerConfig(
        vocab_size=32768,
        vocab_path=str(home / "tokenizer_32k/vocab.json"),
        merges_path=str(home / "tokenizer_32k/merges.json"),
    )

    tokenizer = BPETokenizer(
        tokenizer_config,
        dataset_config,
    )

    output_dir = home / "preprocessed/tokenized/bpe_32768"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 토크나이저 파일 식별
    tokenizer_hashes = {
        "vocab_sha256": sha256_file(tokenizer_config.vocab_path),
        "merges_sha256": sha256_file(tokenizer_config.merges_path),
    }

    for split in ("train", "valid", "test"):
        output_path = output_dir / f"{split}.bin"
        metadata_path = output_dir / f"{split}.meta.json"

        # 기존 파일을 무조건 재사용하지 않음
        if output_path.exists() or metadata_path.exists():
            raise FileExistsError(
                f"기존 결과가 존재합니다: {output_path}\n"
                "토크나이저 혼용 방지를 위해 중단합니다."
            )

        print(f"\n[{split.upper()}] 토큰화 시작", flush=True)
        start = datetime.now()

        documents = load_preprocessed_data(
            split, dataset_config
        )

        token_count = build_tokenized_file(
            documents=documents,
            tokenizer=tokenizer,
            output_path=output_path,
        )

        metadata = {
            "split": split,
            "vocab_size": tokenizer.vocab_size,
            "token_count": token_count,
            "dtype": "uint16",
            **tokenizer_hashes,
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        elapsed = datetime.now() - start

        print(
            f"[{split.upper()}] 완료: "
            f"{token_count:,} tokens, "
            f"소요 시간 {elapsed}",
            flush=True,
        )

    print("\n전체 32K 토큰화 완료!", flush=True)


if __name__ == "__main__":
    main()
