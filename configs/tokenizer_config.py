from dataclasses import dataclass
from pathlib import Path

@dataclass
class PretrainDatasetConfig:
    """
    변수 의미
    dataset_name    학습에 사용할 원본 데이터셋 이름
    subset          데이터셋의 세부 구성
    encoding_type   텍스트 파일 인코딩 방식
    target_bytes    수집할 데이터의 목표 크기
    split_ratios    학습, 검증, 테스트 데이터 분할 비율
    """

    # 데이터셋 설정
    dataset_name: str = "parquet"
    data_files: str = "D:/Deeplearning/data/fineweb-edu/sample/10BT/*.parquet"
    preprocessed_dir: str = "~/preprocessed"
    encoding_type: str = "utf-8"

    # 데이터 수집 및 전처리 설정
    chunk_bytes: int  = 15 * 1024 ** 3  # 15GB
    
    

    # 데이터 분할 비율 (ex) Train 90%, Validation 5%, Test 5%))
    split_ratios: tuple[float, float, float] = (0.90, 0.05, 0.05)

    def __post_init__(self):
        # 데이터 분할 설정값 유효성 검사
        
        # 3개 값 다 넣었는지 확인
        if len(self.split_ratios) != 3:
            raise ValueError("split_ratios는 3개의 비율이어야 합니다.")

        # 양수여야 하는 설정값
        if any(ratio < 0 for ratio in self.split_ratios):
            raise ValueError("분할 비율은 음수일 수 없습니다.")

        # Train, Validation, Test 비율의 합계는 1
        # 부동소수점 계산 오차를 고려하여 1e-9 이하의 차이는 허용
        if abs(sum(self.split_ratios) - 1.0) > 1e-9:
            raise ValueError("분할 비율의 합계는 1이어야 합니다.")

        
@dataclass
class TokenizerConfig:
    vocab_size: int = 32768
    base_vocab_size: int = 256
    
    train_max_bytes: int = 500 * 1024 ** 2  # tokenizer train 500MB
    train_sample_seed: int = 261119

    vocab_path: str = "tokenizer/vocab.json"
    merges_path: str = "tokenizer/merges.json"
    
    pad_token: str = "[PAD]"
    bos_token: str = "[BOS]"
    eos_token: str = "[EOS]"
    cls_token: str = "[CLS]"

    def __post_init__(self):
        self.vocab_path = str(Path(self.vocab_path).expanduser())
        self.merges_path = str(Path(self.merges_path).expanduser())

