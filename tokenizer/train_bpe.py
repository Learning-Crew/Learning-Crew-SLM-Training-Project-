import json
import re
import heapq

from array import array
from pathlib import Path
from itertools import islice
from collections import Counter, defaultdict

from configs.tokenizer_config import TokenizerConfig, PretrainDatasetConfig
from data.download_dataset import load_data

OUTPUT_DIR = Path(__file__).resolve().parent
PRETOKEN_PATTERN = re.compile(r"\s*\S+|\s+")

# 통합 하이퍼파라미터 사용 변수
dataset_config = PretrainDatasetConfig()

def load_pretokens() -> Counter:
    train_data = load_data("train")
    pretoken_counts = Counter()
    document_count = 0
    total_bytes = 0
    encoding_type = dataset_config.encoding_type

    print("Loading entire training dataset...", flush=True)

    for document in train_data:
        text = document.get("text")

        pretokens = (
            match.group(0).encode(encoding_type)
            for match in PRETOKEN_PATTERN.finditer(text)
        )

        pretoken_counts.update(pretokens)

        document_count += 1
        total_bytes += len(text.encode(encoding_type))

    if not pretoken_counts:
        raise ValueError("Training dataset is empty.")

    print(f"Total documents: {document_count:,}")
    print(f"Total bytes: {total_bytes / 1024**3:.3f} GiB")
    print(f"Unique pretokens: {len(pretoken_counts):,}")

    return pretoken_counts

def get_pair_counts(tokens) -> Counter:
    return Counter(
        zip(tokens, islice(tokens, 1, None))
    )

def merge_pair(
    tokens,
    pair: tuple[int, int],
    new_id: int
) -> array:

    result = array("H")
    i = 0

    while i < len(tokens):
        if (
            i + 1 < len(tokens)
            and tokens[i] == pair[0]
            and tokens[i + 1] == pair[1]
        ):
            result.append(new_id)
            i += 2
        else:
            result.append(tokens[i])
            i += 1

    return result


def train_bpe(config: TokenizerConfig):
    special_tokens = (
        config.pad_token,
        config.bos_token,
        config.eos_token,
        config.cls_token,
    )

    max_merges = (
        config.vocab_size - config.base_vocab_size - len(special_tokens)
    )

    if max_merges < 0:
        raise ValueError("Vocabulary size is too small.")        

    print(f"Vocabulary size: {config.vocab_size:,}")
    print(f"Maximum merges: {max_merges:,}")

    pretoken_counts = load_pretokens()

    vocab = [
        bytes([i])
        for i in range(config.base_vocab_size)
    ]

    words = list(pretoken_counts.keys())
    frequencies = list(pretoken_counts.values())

    del pretoken_counts

    pair_freq = Counter()

    pair_words = defaultdict(set)

    print("Building pair frequencies...", flush=True)

    for word_id, tokens in enumerate(words):
        local_counts = get_pair_counts(tokens)

        for pair, count in local_counts.items():
            pair_freq[pair] += (
                count * frequencies[word_id]
            )

            pair_words[pair].add(word_id)

    # 빈도가 높은 Pair부터 선택하는 Heap
    heap = [
        (-freq, pair)
        for pair, freq in pair_freq.items()
    ]

    heapq.heapify(heap)

    merges = []

    print("Training BPE...", flush=True)

    # BPE Merge 반복
    for step in range(max_merges):

        # 유효한 최고 빈도 Pair 선택
        while heap:
            neg_freq, best_pair = heapq.heappop(heap)

            if -neg_freq == pair_freq.get(best_pair, 0):
                break
        else:
            raise RuntimeError(
                f"Insufficient pairs: {len(merges)} merges completed."
            )

        new_id = len(vocab)

        # 새로운 Vocabulary 토큰 생성
        vocab.append(
            vocab[best_pair[0]]
            + vocab[best_pair[1]]
        )

        merges.append(best_pair)

        # 병합 대상이 포함된 Pre-token만 갱신
        affected_words = pair_words.pop(best_pair)

        pair_freq.pop(best_pair)
        changed_pairs = set()

        for word_id in affected_words:
            old_tokens = words[word_id]

            new_tokens = merge_pair(
                old_tokens,
                best_pair,
                new_id
            )

            old_counts = get_pair_counts(old_tokens)
            new_counts = get_pair_counts(new_tokens)

            frequency = frequencies[word_id]

            for pair in old_counts.keys() | new_counts.keys():

                if pair == best_pair:
                    continue

                old_count = old_counts[pair]
                new_count = new_counts[pair]

                # Pair Frequency 증감
                delta = (
                    new_count - old_count
                ) * frequency

                if delta != 0:
                    updated = pair_freq.get(pair, 0) + delta

                    if updated > 0:
                        pair_freq[pair] = updated
                    else:
                        pair_freq.pop(pair, None)

                    changed_pairs.add(pair)

                # Pair 역색인 갱신
                if old_count > 0 and new_count == 0:
                    pair_words[pair].discard(word_id)

                    if not pair_words[pair]:
                        del pair_words[pair]

                elif old_count == 0 and new_count > 0:
                    pair_words[pair].add(word_id)

            words[word_id] = new_tokens

        # 변경된 Pair만 Heap에 추가
        for pair in changed_pairs:
            freq = pair_freq.get(pair, 0)

            if freq > 0:
                heapq.heappush(heap, (-freq, pair))

        # 오래된 Heap 항목 정리
        if len(heap) > 4 * max(len(pair_freq), 1):
            heap = [
                (-freq, pair)
                for pair, freq in pair_freq.items()
            ]
            heapq.heapify(heap)

        if (step + 1) % 500 == 0:
            print(
                f"Merges: {step + 1:,}/{max_merges:,}",
                flush=True
            )

    # 7. 특수 토큰 ID 할당
    special_token_ids = {}

    for token in special_tokens:
        special_token_ids[token] = len(vocab)
        vocab.append(None)

    if len(vocab) != config.vocab_size:
        raise RuntimeError("Vocabulary size mismatch.")

    # 8. Vocabulary 및 Merge 규칙 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vocab_data = {
        "vocab_size": config.vocab_size,
        "pretoken_pattern": PRETOKEN_PATTERN.pattern,
        "tokens_hex": [
            token.hex()
            for token in vocab
            if token is not None
        ],
        "special_tokens": special_token_ids,
    }

    vocab_path = OUTPUT_DIR / "vocab.json"
    merges_path = OUTPUT_DIR / "merges.json"

    with vocab_path.open("w", encoding=dataset_config.encoding_type) as f:
        json.dump(vocab_data, f, ensure_ascii=False)

    with merges_path.open("w", encoding=dataset_config.encoding_type) as f:
        json.dump(merges, f)

    print("\nBPE Training Completed!")
    print(f"Vocabulary size: {len(vocab):,}")
    print(f"Total merges: {len(merges):,}")
    print(f"Vocabulary: {vocab_path}")
    print(f"Merges: {merges_path}")


if __name__ == "__main__":
    config = TokenizerConfig()
    train_bpe(config)
