import json
import hashlib
import unicodedata
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

INPUT_PATH = DATA_DIR / "raw" / "fineWeb_edu_raw.jsonl"
OUTPUT_PATH = DATA_DIR / "processed" / "fineWeb_edu_clean.jsonl"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MIN_CHARS = 200

def clean_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(
        ch for ch in text
        if ch in ("\n", "\t")
        or unicodedata.category(ch) != "Cc"
    )

    return text.strip()

total_count = 0
saved_count = 0
empty_count = 0
short_count = 0
duplicate_count = 0

seen_hashes = set()

with INPUT_PATH.open("r", encoding="utf-8") as src, \
    OUTPUT_PATH.open("w", encoding="utf-8") as dst:

    for line in src:
        total_count += 1

        document = json.loads(line)
        text = document.get("text")

        if not isinstance(text, str) or not text.strip():
            empty_count += 1
            continue

        text = clean_text(text)

        if len(text) < MIN_CHARS:
            short_count += 1
            continue

        text_hash = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

        if text_hash in seen_hashes:
            duplicate_count += 1
            continue

        seen_hashes.add(text_hash)

        document["text"] = text
        dst.write(
            json.dumps(document, ensure_ascii=False) + "\n"
        )

        saved_count += 1

print("PreProcessing completed.")
print("Total documents:", total_count)
print("Empty documents:", empty_count)
print("Short documents:", short_count)
print("Duplicate documents:", duplicate_count)
print("Saved documents:", saved_count)
print("Saved path:", OUTPUT_PATH)
