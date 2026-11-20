
import torch
from torch.utils.data import Dataset
import os
import numpy as np

from pathlib import Path
from tempfile import NamedTemporaryFile

def build_tokenized_file(
    documents,
    tokenizer,
    output_path: str | Path,
) -> int:
    """
    문서를 순차적으로 토큰화하여 uint16 바이너리 파일로 저장한다.
    전체 토큰을 RAM에 보관하지 않는다.
    원래 구조 (전부 다 RAM에 보관) -> 새로운 구조 (디스크에 저장)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    temporary_path = None

    try:
        with NamedTemporaryFile(
            dir=output_path.parent,
            suffix=".tmp",
            delete=False,
        ) as file:

            temporary_path = Path(file.name)

            for document in documents:
                text = document.get("text")

                if not isinstance(text, str) or not text.strip():
                    continue

                ids = tokenizer.encode(
                    text,
                    add_bos=True,
                    add_eos=True,
                )

                if not ids:
                    continue

                if min(ids) < 0 or max(ids) > 65535:
                    raise ValueError("토큰 ID가 Vocabulary 범위를 벗어났습니다.")

                np.asarray(ids, dtype=np.uint16).tofile(file)
                total_tokens += len(ids)

        if total_tokens < 2:
            raise ValueError("저장된 토큰이 부족합니다.")

        os.replace(temporary_path, output_path)
        temporary_path = None

        return total_tokens

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

class TokenizedDataset(Dataset):
    """
    텍스트 데이터를 GPT 학습에 사용할 수 있도록 변환하는 클래스

    주요 역할:
    1. 텍스트를 BPE 토크나이저로 토큰 ID로 변환
    2. 토큰 ID를 하나의 시퀀스로 연결
    3. 지정된 길이만큼 입력과 정답 데이터를 생성
    """

    def __init__(
        self,
        documents=None,
        tokenizer=None,
        seq_len: int = 1024,
        *,
        token_file=None,
        token_count=None,
    ):
        if seq_len <= 0:
            raise ValueError("시퀀스 길이는 0보다 커야 합니다.")

        self.seq_len = seq_len
        self.token_file = None
        self.tokens = None

        # 디스크 기반 Dataset
        if token_file is not None:
            self.token_file = Path(token_file)

            if not self.token_file.is_file():
                raise FileNotFoundError(self.token_file)

            if self.token_file.stat().st_size % 2 != 0:
                raise ValueError("잘못된 uint16 토큰 파일입니다.")

            actual_count = self.token_file.stat().st_size // 2

            if token_count is not None and actual_count != token_count:
                raise ValueError("토큰 개수가 일치하지 않습니다.")

            self.token_count = actual_count

            self.tokens = np.memmap(
                self.token_file,
                dtype=np.uint16,
                mode="r",
            )

        # 기존 메모리 기반 Dataset
        else:
            if documents is None or tokenizer is None:
                raise ValueError("documents와 tokenizer가 필요합니다.")

            self.token_ids = []

            for document in documents:
                text = document.get("text")

                if not isinstance(text, str) or not text:
                    continue

                ids = tokenizer.encode(
                    text,
                    add_bos=True,
                    add_eos=True,
                )

                self.token_ids.extend(ids)

            self.token_count = len(self.token_ids)

        if self.token_count < seq_len + 1:
            raise ValueError("학습 데이터를 생성하기 위한 토큰 수가 부족합니다.")
    
    def __len__(self):
        """
        생성 가능한 학습 샘플의 개수를 반환
        """
        return (self.token_count - 1) // self.seq_len

    def __getitem__(self, index):
        """
        지정된 인덱스에 해당하는 입력과 정답 데이터를 반환
        """

        # 존재하지 않는 인덱스 접근 방지
        if index < 0 or index >= len(self):
            raise IndexError("존재하지 않는 데이터 인덱스입니다.")

        # 현재 샘플의 시작 위치와 끝 위치 계산
        start = index * self.seq_len
        end = start + self.seq_len

        # 디스크 기반 Dataset
        if self.tokens is not None:
            inputs = torch.from_numpy(
                np.array(
                    self.tokens[start:end],
                    dtype=np.int64,
                    copy=True,
                )
            )

            targets = torch.from_numpy(
                np.array(
                    self.tokens[start + 1:end + 1],
                    dtype=np.int64,
                    copy=True,
                )
            )

        # 기존 메모리 기반 Dataset
        else:
            inputs = torch.tensor(
                self.token_ids[start:end],
                dtype=torch.long,
            )

            targets = torch.tensor(
                self.token_ids[start + 1:end + 1],
                dtype=torch.long,
            )

        return inputs, targets
