
import json
import hashlib
import unicodedata

from pathlib import Path
from contextlib import ExitStack

from datasets import load_dataset

from configs.tokenizer_config import PretrainDatasetConfig


SPLIT_RATIOS = {
    "train": 0.90,
    "valid": 0.05,
    "test": 0.05,
}


def load_fineweb(config: PretrainDatasetConfig):
    # 진행도 표시 & 주석 추가 필요
    # 전처리 데이터 공유 불가 (용량문제) > Hugging Face 업로드
    dataset = load_dataset(
        config.dataset_name,
        name=config.subset,
        split="train",
        streaming=True
    )

    ouput_path = Path(__file__).resolve().parent
    total_bytes = 0
    seen_hashes = set()
    counts = { split: 0 for split in SPLIT_RATIOS }

    with ExitStack() as stack:
        outputs = {}

        for split in SPLIT_RATIOS:
            path = ouput_path / f"{split}.json"
            out = stack.enter_context(
                path.open("w", encoding=config.encoding_type)
            )
            out.write("[\n")
            outputs[split] = out

        for document in dataset:
            row = {
                "id": document.get("id"),
                "text": document.get("text"),
                "score": document.get("score")
            }

            text = row["text"]

            if not isinstance(text, str):
                continue

            text = clean_text(text)

            if not text.strip() or len(text.strip()) < config.min_chars:
                continue

            text_bytes = text.encode(config.encoding_type)
            text_hash = hashlib.sha256(text_bytes).digest()

            if text_hash in seen_hashes:
                continue

            seen_hashes.add(text_hash)
            row["text"] = text

            split = assign_split(text_hash)
            out = outputs[split]

            if counts[split] > 0:
                out.write(",\n")

            out.write(
                json.dumps(row, ensure_ascii=False)
            )

            counts[split] += 1
            total_bytes += len(text_bytes)

            if total_bytes >= config.target_bytes:
                break

        for out in outputs.values():
            out.write("\n]\n")

    print(f"finish load dataset: {total_bytes / 1024**3:.2f} GiB")
    print(f"split counts: {counts}")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = "".join(
        char for char in text
        if char in ("\n", "\t")
        or unicodedata.category(char) != "Cc"
    )

    return text


def assign_split(text_hash: bytes) -> str:
    cumulative = 0    
    bucket = int.from_bytes(text_hash[:8], "big") % 10000

    for split, ratio in SPLIT_RATIOS.items():
        cumulative += ratio

        if bucket < round(cumulative * 10000):
            return split

    raise ValueError("Invalid split ratios")


if __name__ == "__main__":
    config = PretrainDatasetConfig()
    load_fineweb(config)
