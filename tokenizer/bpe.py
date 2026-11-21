# load_split_dataset > bpe > bpe_tokenizer 순으로 확인가능합니다!

import json
import re
import heapq
import hydra

from omegaconf import DictConfig, OmegaConf
from array import array
from pathlib import Path
from itertools import islice
from collections import Counter, defaultdict
from contextlib import closing
from random import Random
from loguru import logger

from configs.tokenizer_config import TokenizerConfig, PretrainDatasetConfig
from data.load_preprocessed_data import load_preprocessed_data

# 공백 체크 정규식
PRETOKEN_PATTERN = re.compile(r"\s*\S+|\s+")

def sample_training_texts(
    dataset_config: PretrainDatasetConfig,
    tokenizer_config: TokenizerConfig,
):
    limit = tokenizer_config.train_max_bytes
    generator = Random(tokenizer_config.train_sample_seed)
    selected = []
    selected_bytes = 0
    cutoff = None
    scanned = 0
    oversized = 0
    logger.info(
        f"Sampling train documents up to {limit / 1024**2:.1f} MiB "
        f"(seed={tokenizer_config.train_sample_seed}); scanning all train shards...",
    )

    with closing(load_preprocessed_data("train", dataset_config)) as documents:
        for document in documents:
            text = document["text"]
            if not text.strip():
                continue
            scanned += 1
            if scanned % 100_000 == 0:
                logger.info(f"Scanned train documents: {scanned:,}")
            size = len(text.encode(dataset_config.encoding_type))
            if size > limit:
                oversized += 1
                continue

            priority = (generator.getrandbits(128), scanned)
            if cutoff is not None and priority >= cutoff:
                continue
            
            heapq.heappush(selected, (-priority[0], -priority[1], size, text))
            selected_bytes += size
            while selected_bytes > limit:
                removed = heapq.heappop(selected)
                selected_bytes -= removed[2]
                cutoff = (-removed[0], -removed[1])

    logger.info(
        f"Selected {len(selected):,} / {scanned:,} documents, "
        f"{selected_bytes / 1024**2:.3f} MiB; oversized skipped: {oversized:,}",
    )
    
    while selected:
        yield heapq.heappop(selected)[3]


def load_pretokens(
    dataset_config: PretrainDatasetConfig,
    tokenizer_config: TokenizerConfig,
) -> Counter:
    pretoken_counts = Counter()
    document_count = 0
    total_bytes = 0
    encoding_type = dataset_config.encoding_type

    with closing(sample_training_texts(dataset_config, tokenizer_config)) as samples:
        for text in samples:
            document_bytes = len(text.encode(encoding_type))
            pretokens = (
                # 정규식에 따른 텍스트 분리 > byte 변환
                match.group(0).encode(encoding_type)
                for match in PRETOKEN_PATTERN.finditer(text)
            )
            # 중복 텍스트 제거
            pretoken_counts.update(pretokens)
            document_count += 1
            total_bytes += document_bytes

    if not pretoken_counts:
        raise ValueError("학습할 텍스트가 없습니다. train 파일과 train_max_bytes를 확인하세요.")

    logger.info(
        f"Pretoken counting completed | documents={document_count:,} | "
        f"size={total_bytes / 1024**3:.3f} GiB | "
        f"unique_pretokens={len(pretoken_counts):,}"
    )

    return pretoken_counts

def get_pair_counts(tokens) -> Counter:
    return Counter(zip(tokens, islice(tokens, 1, None)))

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


def train_bpe(
    config: TokenizerConfig,
    dataset_config: PretrainDatasetConfig,
):
    special_tokens = (
        config.pad_token,
        config.bos_token,
        config.eos_token,
        config.cls_token,
    )

    max_merges = (config.vocab_size - config.base_vocab_size - len(special_tokens))

    if max_merges < 0:
        raise ValueError("Vocabulary size is too small.")        

    logger.info(
        f"BPE configuration | vocab_size={config.vocab_size:,} | "
        f"max_merges={max_merges:,}"
    )

    pretoken_counts = load_pretokens(dataset_config, config)

    vocab = [bytes([i]) for i in range(config.base_vocab_size)]

    words = list(pretoken_counts.keys())
    frequencies = list(pretoken_counts.values())

    del pretoken_counts

    pair_freq = Counter()
    pair_words = defaultdict(set)

    logger.info("Building pair frequencies...")

    for word_id, tokens in enumerate(words):
        local_counts = get_pair_counts(tokens)

        for pair, count in local_counts.items():
            pair_freq[pair] += ( count * frequencies[word_id] )
            pair_words[pair].add(word_id)

    heap = [(-freq, pair) for pair, freq in pair_freq.items()]

    heapq.heapify(heap)

    merges = []

    logger.info("Training BPE...")

    for step in range(max_merges):

        while heap:
            neg_freq, best_pair = heapq.heappop(heap)

            if -neg_freq == pair_freq.get(best_pair, 0):
                break
        else:
            raise RuntimeError(
                f"Insufficient pairs: {len(merges)} merges completed."
            )

        new_id = len(vocab)
        vocab.append(vocab[best_pair[0]] + vocab[best_pair[1]])
        merges.append(best_pair)

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

                delta = (new_count - old_count) * frequency

                if delta != 0:
                    updated = pair_freq.get(pair, 0) + delta

                    if updated > 0:
                        pair_freq[pair] = updated
                    else:
                        pair_freq.pop(pair, None)

                    changed_pairs.add(pair)

                if old_count > 0 and new_count == 0:
                    pair_words[pair].discard(word_id)

                    if not pair_words[pair]:
                        del pair_words[pair]

                elif old_count == 0 and new_count > 0:
                    pair_words[pair].add(word_id)

            words[word_id] = new_tokens

        for pair in changed_pairs:
            freq = pair_freq.get(pair, 0)

            if freq > 0:
                heapq.heappush(heap, (-freq, pair))

        if len(heap) > 4 * max(len(pair_freq), 1):
            heap = [(-freq, pair) for pair, freq in pair_freq.items()]
            heapq.heapify(heap)

        if (step + 1) % 500 == 0:
            logger.info(f"Merges: {step + 1:,}/{max_merges:,}")

    special_token_ids = {}

    for token in special_tokens:
        special_token_ids[token] = len(vocab)
        vocab.append(None)

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

    
    Path(config.vocab_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config.merges_path).parent.mkdir(parents=True, exist_ok=True)

    with open(config.vocab_path, "w", encoding=dataset_config.encoding_type) as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=4)

    with open(config.merges_path, "w", encoding=dataset_config.encoding_type) as f:
        json.dump(merges, f, indent=4)

    logger.success(
        f"BPE training completed | vocab_size={len(vocab):,} | "
        f"merges={len(merges):,} | "
        f"vocab_path={config.vocab_path} | "
        f"merges_path={config.merges_path}"
    )


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="experiment",
)
def main(config: DictConfig) -> None:
    tokenizer_config = TokenizerConfig(
        **OmegaConf.to_container(config.tokenizer, resolve=True)
    )
    dataset_config = PretrainDatasetConfig(
        **OmegaConf.to_container(config.dataset, resolve=True)
    )

    train_bpe(tokenizer_config, dataset_config)


if __name__ == "__main__":
    main()
