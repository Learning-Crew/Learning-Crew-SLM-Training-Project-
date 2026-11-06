
import json
import re
import heapq
from pathlib import Path
from collections import Counter, defaultdict


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.jsonl"

OUTPUT_DIR = PROJECT_ROOT / "tokenizer" / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "bpe_8192.json"

VOCAB_SIZE = 8192
EOS_ID = VOCAB_SIZE - 1
MAX_MERGES = VOCAB_SIZE - 256 - 1

PRETOKEN_PATTERN = r"\s*\S+|\s+"
pattern = re.compile(PRETOKEN_PATTERN)


def load_pretoken_counts(path):
    counts = Counter()
    document_count = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            document = json.loads(line)
            text = document["text"]

            pieces = pattern.findall(text)
            counts.update(pieces)

            document_count += 1

            if document_count % 1000 == 0:
                print(f"READ {document_count:,} documents...", flush=True)

    print("Total documents:", document_count)
    print("Unique pieces:", len(counts))

    return counts

def get_pair_counts(ids):
    return Counter(zip(ids, ids[1:]))

def merge(ids, pair, new_id):
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

    return tuple(result)

def train_bpe(pretoken_counts):
    words = []
    frequencies = []
    pair_counts = Counter()
    pair_to_words = defaultdict(set)

    for piece, frequency in pretoken_counts.items():
        ids = tuple(piece.encode("utf-8"))

        word_id = len(words)
        words.append(ids)
        frequencies.append(frequency)

        local_counts = get_pair_counts(ids)

        for pair, count in local_counts.items():
            pair_counts[pair] += count * frequency
            pair_to_words[pair].add(word_id)

    heap = [
        (-count, pair)
        for pair, count in pair_counts.items()
    ]
    heapq.heapify(heap)

    merges = []

    print("Start training BPE...", flush=True)

    for step in range(MAX_MERGES):
        while heap:
            negative_count, pair = heap[0]

            if pair_counts.get(pair, 0) == -negative_count:
                break

            heapq.heappop(heap)

        if not heap:
            break

        negative_count, best_pair = heapq.heappop(heap)
        best_count = -negative_count

        if best_count < 2:
            print("No more frequent pairs to merge.")
            break

        new_id = 256 + len(merges)
        merges.append(best_pair)
        affected_ids = list(pair_to_words.get(best_pair, ()))
        touched_pairs = set()

        for word_id in affected_ids:
            old_ids = words[word_id]
            new_ids = merge(old_ids, best_pair, new_id)
            old_counts = get_pair_counts(old_ids)
            new_counts = get_pair_counts(new_ids)
            old_keys = old_counts.keys()
            new_keys = new_counts.keys()


            for pair in old_keys - new_keys:
                pair_to_words[pair].discard(word_id)

                if not pair_to_words[pair]:
                    del pair_to_words[pair]

            for pair in new_keys - old_keys:
                pair_to_words[pair].add(word_id)

            for pair in old_keys | new_keys:
                difference = (
                    new_counts.get(pair, 0)
                    - old_counts.get(pair, 0)
                ) * frequencies[word_id]

                if difference == 0:
                    continue

                updated_count = pair_counts.get(pair, 0) + difference

                if updated_count <= 0:
                    pair_counts.pop(pair, None)
                else:
                    pair_counts[pair] = updated_count

                touched_pairs.add(pair)

            words[word_id] = new_ids

        for pair in touched_pairs:
            count = pair_counts.get(pair, 0)

            if count > 0:
                heapq.heappush(heap, (-count, pair))

        if (step + 1) % 100 == 0:
            print(
                f"Merge {step + 1:,}/{MAX_MERGES:,} completed "
                f"| Current best pair frequency: {best_count:,}",
                flush=True
            )

    return merges


def build_vocab(merges):
    vocab = {
        i: bytes([i])
        for i in range(256)
    }

    for index, (left, right) in enumerate(merges):
        new_id = 256 + index

        vocab[new_id] = vocab[left] + vocab[right]

    return vocab

def save_model(merges, path):
    vocab = build_vocab(merges)

    model = {
        "vocab_size": VOCAB_SIZE,
        "eos_id": EOS_ID,
        "pretoken_pattern": PRETOKEN_PATTERN,
        "merges": [
            [left, right]
            for left, right in merges
        ],
        "vocab_hex": {
            str(token_id): token_bytes.hex()
            for token_id, token_bytes in vocab.items()
        }
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False)

    print("BPE training completed!")
    print("Base byte tokens:", 256)
    print("Number of learned merges:", len(merges))
    print("Number of normal tokens:", len(vocab))
    print("EOS Token ID:", EOS_ID)
    print("Saved to:", path)

if __name__ == "__main__":
    pretoken_counts = load_pretoken_counts(TRAIN_PATH)

    merges = train_bpe(pretoken_counts)

    save_model(merges, OUTPUT_PATH)
