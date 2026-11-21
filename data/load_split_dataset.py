
import json
import re
import hashlib
import unicodedata
import sqlite3
import hydra

from omegaconf import DictConfig, OmegaConf
from loguru import logger
from pathlib import Path
from contextlib import ExitStack, closing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import islice

from datasets import load_dataset

from configs.tokenizer_config import PretrainDatasetConfig

# 병렬처리 대응
NUM_WORKERS = 4
BATCH_SIZE = 128

def clean_text(text: str) -> str:

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = "".join(
        char for char in text
        if char in ("\n", "\t")
        or unicodedata.category(char) != "Cc"
    )

    return text


def assign_split(
    text_hash: bytes,
    split_ratios: dict[str, float]
) -> str:

    cumulative = 0
    bucket = int.from_bytes(text_hash[:8], "big") % 10000

    for split, ratio in split_ratios.items():
        cumulative += ratio

        if bucket < round(cumulative * 10000):
            return split

    raise ValueError("Invalid split ratios")



def preprocess_document(document, encoding_type):

    row = {
        "id": document.get("id"),
        "text": document.get("text"),
    }

    text = row["text"]

    if not isinstance(text, str):
        return None

    text = clean_text(text)

    if not text.strip():
        return None

    text_bytes = text.encode(encoding_type)
    text_hash = hashlib.sha256(text_bytes).digest()

    row["text"] = text

    return row, text_hash, len(text_bytes)



def preprocess_dataset(dataset, config):

    worker = partial(
        preprocess_document,
        encoding_type=config.encoding_type
    )

    documents = iter(dataset)

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
    ) as executor:

        while True:
            batch = list(
                islice(documents, BATCH_SIZE)
            )

            if not batch:
                break

            # 여러 프로세스에서 전처리
            results = executor.map(
                worker,
                batch,
                chunksize=8
            )

            for result in results:

                if result is not None:
                    yield result



def remove_stale_shards(output_dir: Path, last_chunk_index: int) -> int:
    output_dir = output_dir.resolve()
    removed = 0
    for path in output_dir.iterdir():
        match = re.fullmatch(r"(?:train|valid|test)_([0-9]+)\.json", path.name)
        if match is None or int(match.group(1)) <= last_chunk_index:
            continue
        
        if path.is_file() or path.is_symlink():
            path.unlink()
            removed += 1
    return removed


def load_fineweb(config: PretrainDatasetConfig, max_documents: int | None = None):

    split_ratios = {
        "train": config.split_ratios[0],
        "valid": config.split_ratios[1],
        "test": config.split_ratios[2],
    }

    dataset = load_dataset(
        config.dataset_name,
        data_files=config.data_files,
        split="train",
        streaming=True
    )
    if max_documents is not None:
        dataset = dataset.take(max_documents)
        logger.info(f"테스트 모드: 최대 {max_documents:,}개 원본 문서 처리")
    
    root_path = Path(config.preprocessed_dir).expanduser()
    root_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Preprocessing output directory: {root_path}")
    db_path = root_path / "seen_hashes.db"
    total_bytes = 0
    total_docs = 0
    chunk_bytes = 0
    chunk_index = 0
    counts = { split: 0 for split in split_ratios }
    chunk_counts = { split: 0 for split in split_ratios }

    with ExitStack() as stack:
        conn = stack.enter_context(
            closing(sqlite3.connect(db_path))
        )

        # 기존 해시 테이블 초기화 후 재생성
        conn.execute("PRAGMA cache_size = -32768")        
        conn.execute("DROP TABLE IF EXISTS seen_hashes")
        conn.execute(
            """
            CREATE TABLE seen_hashes (
                hash BLOB PRIMARY KEY
            ) WITHOUT ROWID
            """
        )

        conn.commit()

        file_stack = stack.enter_context(ExitStack())
        outputs = {}

        for row, text_hash, document_bytes in preprocess_dataset(
            dataset, config
        ):

            cursor = conn.execute(
                "INSERT OR IGNORE INTO seen_hashes (hash) VALUES (?)",
                (text_hash,)
            )

            if cursor.rowcount == 0:
                continue

            # chunk_bytes를 초과할 경우 현재 파일 세트 종료
            if (
                chunk_bytes > 0
                and chunk_bytes + document_bytes > config.chunk_bytes
            ):

                for out in outputs.values():
                    out.write("\n]\n")

                file_stack.close()

                logger.info(
                    f"Chunk {chunk_index:03d} completed: "
                    f"{chunk_bytes / 1024**2:.2f} MiB",
                )

                chunk_index += 1
                chunk_bytes = 0
                chunk_counts = { split: 0 for split in split_ratios }

                outputs = {}

            if not outputs:
                for split in split_ratios:
                    path = (root_path / f"{split}_{chunk_index:03d}.json")

                    out = file_stack.enter_context(
                        path.open("w", encoding=config.encoding_type)
                    )

                    out.write("[\n")
                    outputs[split] = out

                logger.info(f"Start chunk {chunk_index:03d}")

            split = assign_split(text_hash, split_ratios)
            out = outputs[split]

            if chunk_counts[split] > 0:
                out.write(",\n")

            out.write(json.dumps(row, ensure_ascii=False))

            counts[split] += 1
            chunk_counts[split] += 1
            total_docs += 1
            total_bytes += document_bytes
            chunk_bytes += document_bytes

            # 10,000개 문서마다 DB 저장 및 진행도 출력
            if total_docs % 10000 == 0:
                conn.commit()

                logger.info(
                    f"Chunk {chunk_index:03d}: "
                    f"{chunk_bytes / 1024**3:.2f} / {config.chunk_bytes / 1024**3:.2f} GiB | "
                    f"Total: {total_bytes / 1024**3:.2f} GiB | "
                    f"Documents: {total_docs:,}",
                )

        for out in outputs.values():
            out.write("\n]\n")

        file_stack.close()
        conn.commit()

    
    
    last_chunk_index = chunk_index if total_docs > 0 else -1
    removed = remove_stale_shards(root_path, last_chunk_index)
    logger.info(f"Removed stale shard files: {removed}")

    logger.success(
        f"Preprocessing completed | "
        f"size={total_bytes / 1024**2:.2f} MiB | "
        f"documents={total_docs:,} | "
        f"splits={counts}"
    )



@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="experiment",
)
def main(config: DictConfig) -> None:
    dataset_config = PretrainDatasetConfig(
        **OmegaConf.to_container(
            config.dataset,
            resolve=True,
        )
    )

    load_fineweb(dataset_config)


if __name__ == "__main__":
    main()

