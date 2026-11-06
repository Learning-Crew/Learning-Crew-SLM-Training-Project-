import json
import hashlib
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

INPUT_PATH = DATA_DIR / "processed" / "fineWeb_edu_clean.jsonl"
OUTPUT_DIR = DATA_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_split(document_id):
    hash_value = hashlib.sha256(
        str(document_id).encode("utf-8")
    ).digest()

    value = int.from_bytes(hash_value[:8], "big") % 100

    if value < 90:
        return "train"
    elif value < 95:
        return "validation"
    else:
        return "test"

counts = {
    "train": 0,
    "validation": 0,
    "test": 0
}

with (
    INPUT_PATH.open("r", encoding="utf-8") as src,
    (OUTPUT_DIR / "train.jsonl").open("w", encoding="utf-8") as train,
    (OUTPUT_DIR / "validation.jsonl").open("w", encoding="utf-8") as validation,
    (OUTPUT_DIR / "test.jsonl").open("w", encoding="utf-8") as test
):

    files = {
        "train": train,
        "validation": validation,
        "test": test
    }

    for line in src:
        if not line.strip():
            continue

        document = json.loads(line)
        split = get_split(document["id"])
        files[split].write(
            json.dumps(document, ensure_ascii=False) + "\n"
        )
        
        counts[split] += 1

total = sum(counts.values())

for split, count in counts.items():
    ratio = count / total * 100 if total else 0
    print(f"{split}: {count} ({ratio:.2f}%)")

print("total documents:", total)