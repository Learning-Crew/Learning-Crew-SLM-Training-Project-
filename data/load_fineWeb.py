import json
import requests
import pyarrow.parquet as pq
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_PATH = RAW_DIR / "fineWeb_edu.parquet"
OUTPUT_PATH = RAW_DIR / "fineWeb_edu_raw.jsonl"

URL = (
    "https://huggingface.co/datasets/"
    "HuggingFaceFW/fineweb-edu/resolve/"
    "7938818d82a49662629895c862dd52a3b23d213e/"
    "sample/10BT/013_00000.parquet"
)

TARGET_BYTES = 100 * 1024 * 1024

if not PARQUET_PATH.exists():
    print("Start Downloading...")

    with requests.get(
        URL,
        stream=True,
        timeout=(30, 120)
    ) as response:

        response.raise_for_status()

        temp_path = PARQUET_PATH.with_suffix(".parquet.part")

        with temp_path.open("wb") as f:
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    f.write(chunk)

    temp_path.replace(PARQUET_PATH)
    print("Finish DownLoading.")

else:
    print("File already exists.")

parquet_file = pq.ParquetFile(PARQUET_PATH)

total_bytes = 0
document_count = 0

with OUTPUT_PATH.open("w", encoding="utf-8") as output:
    for batch in parquet_file.iter_batches(
        batch_size=512,
        columns=["id", "text", "score"]
    ):

        for row in batch.to_pylist():
            text = row["text"]

            if not isinstance(text, str):
                continue

            document = {
                "id": row["id"],
                "text": text,
                "score": row["score"]
            }

            output.write(
                json.dumps(document, ensure_ascii=False) + "\n"
            )

            total_bytes += len(text.encode("utf-8"))
            document_count += 1

            if total_bytes >= TARGET_BYTES:
                break

        if total_bytes >= TARGET_BYTES:
            break

print("Finish converting parquet to json.")
#print("document_count:", document_count)
#print("text bytes:", total_bytes)
#print("output path:", OUTPUT_PATH)