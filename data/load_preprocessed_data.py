from collections.abc import Iterator
from itertools import chain
from pathlib import Path

import ijson

from configs.tokenizer_config import PretrainDatasetConfig


def load_preprocessed_data(split: str, config: PretrainDatasetConfig) -> Iterator[dict]:
    """
    config.preprocessed_dir의 전처리된 JSON 분할 파일을 순서대로 읽습니다.

    Args:
        split: "train", "valid", "test"
    """
    if split not in ("train", "valid", "test"):
        raise ValueError("split은 train, valid, test 중 하나여야 합니다.")
    data_dir = Path(config.preprocessed_dir).expanduser()
    files = sorted(
        (
            path for path in data_dir.glob(f"{split}_*.json")
            if path.is_file() and path.stem[len(split) + 1:].isdecimal()
        ),
        key=lambda path: (int(path.stem[len(split) + 1:]), path.name),
    )
    if not files:
        raise FileNotFoundError(
            f"전처리 파일이 없습니다: {data_dir / (split + '_*.json')}. "
            "먼저 data.load_split_dataset으로 전처리하거나 preprocessed_dir를 확인하세요."
        )

    for path in files:
        try:
            with path.open("r", encoding=config.encoding_type) as source:
                events = ijson.parse(source, use_float=True)
                first = next(events, None)
                if first != ("", "start_array", None):
                    raise ValueError(f"{path}: 전처리 파일은 JSON 배열이어야 합니다.")
                for index, document in enumerate(
                    ijson.items(chain((first,), events), "item"), start=1
                ):
                    if not isinstance(document, dict) or not isinstance(document.get("text"), str):
                        raise ValueError(f"{path}: {index}번째 문서에 문자열 text가 없습니다.")
                    yield document
        except (ijson.JSONError, UnicodeError) as error:
            raise ValueError(f"{path}: 전처리 JSON을 읽을 수 없습니다: {error}") from error
