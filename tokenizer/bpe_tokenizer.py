
import json
import re

from pathlib import Path
from functools import lru_cache

from configs.tokenizer_config import TokenizerConfig, PretrainDatasetConfig

config = TokenizerConfig()
dataset_config = PretrainDatasetConfig()

class BPETokenizer:

    def __init__(self):
        with open(config.vocab_path, "r", encoding=dataset_config.encoding_type) as f:
            vocab_data = json.load(f)

        with open(config.merges_path, "r", encoding=dataset_config.encoding_type) as f:
            merges = json.load(f)

        self.vocab_size = vocab_data["vocab_size"]
        self.pretoken_pattern = re.compile(vocab_data["pretoken_pattern"])

        self.vocab = [
            bytes.fromhex(token)
            for token in vocab_data["tokens_hex"]
        ]


        self.special_tokens = vocab_data["special_tokens"]
        self.pad_id = self.special_tokens[config.pad_token]
        self.bos_id = self.special_tokens[config.bos_token]
        self.eos_id = self.special_tokens[config.eos_token]
        self.cls_id = self.special_tokens[config.cls_token]

        self.id_to_special = {
            token_id: token for token, token_id in self.special_tokens.items()
        }

        self.merge_ids = {
            tuple(pair): 256 + rank for rank, pair in enumerate(merges)
        }


        if len(self.vocab) + len(self.special_tokens) != self.vocab_size:
            raise ValueError("Vocabulary size mismatch.")

        if len(merges) != len(self.vocab) - 256:
            raise ValueError("Merge count mismatch.")


    @lru_cache(maxsize=50_000)
    def encode_piece(self, piece: bytes) -> tuple[int, ...]:
        tokens = list(piece)

        while len(tokens) > 1:
            best_pair = min(
                zip(tokens, tokens[1:]),
                key=lambda pair: self.merge_ids.get(pair, self.vocab_size)
            )

            new_id = self.merge_ids.get(best_pair)

            if new_id is None:
                break

            merged = []
            i = 0

            while i < len(tokens):

                if (
                    i + 1 < len(tokens)
                    and tokens[i] == best_pair[0]
                    and tokens[i + 1] == best_pair[1]
                ):
                    merged.append(new_id)
                    i += 2

                else:
                    merged.append(tokens[i])
                    i += 1

            tokens = merged

        return tuple(tokens)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False
    ) -> list[int]:

        token_ids = []

        if add_bos:
            token_ids.append(self.bos_id)

        for match in self.pretoken_pattern.finditer(text):
            piece = match.group(0).encode(dataset_config.encoding_type)
            token_ids.extend(self.encode_piece(piece))

        if add_eos:
            token_ids.append(self.eos_id)

        return token_ids


    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = True,
        errors: str = "strict"
    ) -> str:

        text_bytes = bytearray()

        for token_id in token_ids:
            if token_id in self.id_to_special:
                if not skip_special_tokens:
                    token = self.id_to_special[token_id]
                    text_bytes.extend(token.encode(self.encoding_type))

                continue

            if not 0 <= token_id < len(self.vocab):
                raise ValueError(f"Invalid token ID: {token_id}")

            text_bytes.extend(self.vocab[token_id])

        return text_bytes.decode(dataset_config.encoding_type, errors=errors)

"""
if __name__ == "__main__":

    tokenizer = BPETokenizer()

    text = "Hello world! I am learning BPE."

    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)

    print("Original:", repr(text))
    print("Token IDs:", token_ids)
    print("Decoded:", repr(decoded))

    print("Token count:", len(token_ids))
    print("Vocabulary size:", tokenizer.vocab_size)

    assert text == decoded

    print("Round-trip test passed!")
"""