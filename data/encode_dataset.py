
import json
import sys
from array import array
from pathlib import Path

from tokenizer.bpe_tokenizer import BPETokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "tokenized" / "bpe_8192"

MODEL_PATH = (
    PROJECT_ROOT / "tokenizer" / "artifacts" / "bpe_8192.json"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FLUSH_TOKENS = 65536


def flush_tokens(out, buffer):
    if not buffer:
        return

    if sys.byteorder == "little":
        out.write(buffer.tobytes())
    else:
        temp = array("H", buffer)
        temp.byteswap()
        out.write(temp.tobytes())

    buffer.clear()

def encode_file(tokenizer, input_path, output_path):

    document_count = 0
    token_count = 0
    text_token_count = 0

    buffer = array("H")

    temp_path = output_path.with_suffix(".bin.part")

    with (
        input_path.open("r", encoding="utf-8") as src,
        temp_path.open("wb") as out
    ):
        for line in src:
            if not line.strip():
                continue

            document = json.loads(line)
            text = document["text"]          
            ids = tokenizer.encode(text)
     
            if any(
                token_id < 0 or token_id >= tokenizer.eos_id
                for token_id in ids
            ):
                raise ValueError("Token ID error")

            buffer.extend(ids)
            buffer.append(tokenizer.eos_id)

            document_count += 1
            text_token_count += len(ids)
            token_count += len(ids) + 1

            if len(buffer) >= FLUSH_TOKENS:
                flush_tokens(out, buffer)

            if document_count % 1000 == 0:
                print(
                    f"document {document_count:,} encoded, total tokens: {token_count:,}",
                    flush=True
                )

        flush_tokens(out, buffer)

    expected_bytes = token_count * 2
    actual_bytes = temp_path.stat().st_size

    if actual_bytes != expected_bytes:
        raise ValueError("binary file size mismatch")

    temp_path.replace(output_path)

    return {
        "documents": document_count,
        "text_tokens": text_token_count,
        "tokens_with_eos": token_count,
        "file": output_path.name
    }

def main():

    tokenizer = BPETokenizer(MODEL_PATH)

    if tokenizer.vocab_size > 65536:
        raise ValueError("not enough uint16 for Vocabulary")

    if tokenizer.eos_id in tokenizer.vocab:
        raise ValueError("Error EOS ID collision with normal token ID")

    print("Vocabulary Size:", tokenizer.vocab_size)
    print("EOS ID:", tokenizer.eos_id)

    metadata = {
        "tokenizer": "bpe",
        "vocab_size": tokenizer.vocab_size,
        "eos_id": tokenizer.eos_id,
        "dtype": "uint16",
        "byte_order": "little",
        "splits": {}
    }

    for split in ["train", "validation", "test"]:

        input_path = INPUT_DIR / f"{split}.jsonl"
        output_path = OUTPUT_DIR / f"{split}.bin"

        print(f"\n===== {split} Encoding Started =====")

        result = encode_file(
            tokenizer,
            input_path,
            output_path
        )

        metadata["splits"][split] = result

        print("Document Count:", result["documents"])
        print("Token Count (with EOS):", result["tokens_with_eos"])
        print("Save Path:", output_path)

    metadata_path = OUTPUT_DIR / "metadata.json"

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nTotal Data Encoding Completed!")


if __name__ == "__main__":
    main()
