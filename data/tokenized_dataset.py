
import torch
from torch.utils.data import Dataset


class TokenizedDataset(Dataset):
    """
    텍스트 데이터를 GPT 학습에 사용할 수 있도록 변환하는 클래스

    주요 역할:
    1. 텍스트를 BPE 토크나이저로 토큰 ID로 변환
    2. 토큰 ID를 하나의 시퀀스로 연결
    3. 지정된 길이만큼 입력과 정답 데이터를 생성
    """

    def __init__(self, documents, tokenizer, seq_len: int):
        # 시퀀스 길이는 0이상
        if seq_len <= 0:
            raise ValueError("시퀀스 길이는 0보다 커야 합니다.")

        self.seq_len = seq_len
        self.token_ids = []

        # 각 문서마다 토큰화
        for document in documents:
            text = document.get("text")

            # 비어있으면 스킾
            if not isinstance(text, str) or not text:
                continue

            # bpe_tokenizer를 사용하여 텍스트를 토큰 ID로 변환
            # BOS: 문서 시작
            # EOS: 문서 끝
            ids = tokenizer.encode(
                text,
                add_bos=True,
                add_eos=True,
            )

            # 각 문서의 토큰 ID 연결
            self.token_ids.extend(ids)

        # 입력과 정답을 만들려면 seq_len + 1개의 토큰이 필요
        if len(self.token_ids) < seq_len + 1:
            raise ValueError(
                "학습 데이터를 생성하기 위한 토큰 수가 부족합니다."
            )

    def __len__(self):
        """
        생성 가능한 학습 샘플의 개수를 반환
        """
        return (len(self.token_ids) - 1) // self.seq_len

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

        # 모델에 입력할 토큰 ID
        inputs = torch.tensor(
            self.token_ids[start:end],
            dtype=torch.long,
        )

        # 정답은 입력보다 한 칸 뒤의 토큰 ID
        # 우리 모델은 다음 토큰을 예측하도록 학습하기 때문
        targets = torch.tensor(
            self.token_ids[start + 1:end + 1],
            dtype=torch.long,
        )

        return inputs, targets
