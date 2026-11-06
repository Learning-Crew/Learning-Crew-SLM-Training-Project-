
import json
import re
from pathlib import Path
from functools import lru_cache


class BPETokenizer:

    def __init__(self, model_path):

        with open(model_path, "r", encoding="utf-8") as f:
            model = json.load(f)

        self.vocab_size = model["vocab_size"]
        self.eos_id = model["eos_id"]
        self.pattern = re.compile(model["pretoken_pattern"])
        self.merge_ids = {
            tuple(pair): 256 + rank
            for rank, pair in enumerate(model["merges"])
        }

        self.vocab = {
            int(token_id): bytes.fromhex(hex_string)
            for token_id, hex_string in model["vocab_hex"].items()
        }

    def merge(self, ids, pair, new_id):
        result = []
        i = 0

        while i < len(ids):
            if (
                i + 1 < len(ids)
                and (ids[i], ids[i + 1]) == pair
            ):
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1

        return result

    @lru_cache(maxsize=50000)
    def encode_piece(self, piece):
        ids = list(piece.encode("utf-8"))

        while len(ids) > 1:
            best_pair = None
            best_id = None

            for pair in zip(ids, ids[1:]):
                new_id = self.merge_ids.get(pair)

                if new_id is None:
                    continue

                if best_id is None or new_id < best_id:
                    best_pair = pair
                    best_id = new_id

            if best_pair is None:
                break

            ids = self.merge(ids, best_pair, best_id)

        return tuple(ids)

    def encode(self, text, add_eos=False):
        token_ids = []
        pieces = self.pattern.findall(text)

        for piece in pieces:
            token_ids.extend(self.encode_piece(piece))

        if add_eos:
            token_ids.append(self.eos_id)

        return token_ids

    def decode(self, token_ids):
        result = bytearray()

        for token_id in token_ids:
            if token_id == self.eos_id:
                continue

            if token_id not in self.vocab:
                raise ValueError(
                    f"Unknown Token ID: {token_id}"
                )

            result.extend(self.vocab[token_id])

        return result.decode("utf-8")

if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    MODEL_PATH = (
        PROJECT_ROOT / "tokenizer" / "artifacts" / "bpe_8192.json"
    )

    VALID_PATH = (
        PROJECT_ROOT / "data" / "processed" / "validation.jsonl"
    )

    tokenizer = BPETokenizer(MODEL_PATH)

    print("Vocabulary Size:", tokenizer.vocab_size)
    print("Learned Merge Rules:", len(tokenizer.merge_ids))
    print("EOS Token ID:", tokenizer.eos_id)
    print("\n===== Sample Encoding/Decoding =====")

    document_count = 0
    total_bytes = 0
    total_tokens = 0

    with VALID_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            document = json.loads(line)
            text = document["text"]

            ids = tokenizer.encode(text)
            restored = tokenizer.decode(ids)

            if restored != text:
                raise ValueError(
                    f"Document restoration failed: {document.get('id')}"
                )

            document_count += 1
            total_bytes += len(text.encode("utf-8"))
            total_tokens += len(ids)

    print("document count:", document_count)
    print("total bytes:", total_bytes)
    print("total tokens:", total_tokens)

    if total_tokens > 0:
        print(
            "Bytes per Token:",
            round(total_bytes / total_tokens, 4)
        )

    print("Finished sample encoding/decoding test.")
