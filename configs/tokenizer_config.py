from dataclasses import dataclass

@dataclass
class PretrainDatasetConfig:
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    subset: str = "sample-10BT"
    encoding_type: str = "utf-8"

    target_bytes = 2 * 1024 ** 3    # 2GB
    min_chars = 100
    max_chars = 100_000


@dataclass
class TokenizerConfig:
    vocab_size: int = 32768
    base_vocab_size: int = 256
    
    pad_token: str = "[PAD]"
    bos_token: str = "[BOS]"
    eos_token: str = "[EOS]"
    cls_token: str = "[CLS]"