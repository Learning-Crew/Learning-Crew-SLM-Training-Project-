
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    하이퍼 파라미터(적절히 수정!)

    변수 의미
    learning_rate   모델이 한 번에 얼마나 크게 가중치를 수정할지
    batch_size	    한 번의 학습에 사용하는 데이터 묶음 수
    max_steps	    가중치를 업데이트할 최대 횟수
    weight_decay	가중치가 지나치게 커지는 것을 억제하는 정규화 설정
    seed             epoch별 데이터 순서를 정하는 시드 (재현성을 위해 필요)
    checkpoint_every 체크포인트를 저장할 스텝 간격 (너무 자주 저장하면 용량이 커질 수 있습니다!)
    """

    learning_rate: float = 3e-4
    batch_size: int = 8
    max_steps: int = 10000
    weight_decay: float = 0.1
    seed: int = 42
    checkpoint_every: int = 500     #이건 학습을 해보고 얼마나 저장되는지 봐야할 것 같아요!

    def __post_init__(self):
        # 모델 설정 값 오류 잡기
        
        # 학습률, 배치 크기, 학습 횟수는 0 이상 
        if self.learning_rate <= 0:
            raise ValueError("학습률은 0보다 커야 합니다.")

        if self.batch_size <= 0:
            raise ValueError("배치 크기는 0보다 커야 합니다.")

        if self.max_steps <= 0:
            raise ValueError("학습 횟수는 0보다 커야 합니다.")

        # Weight Decay는 0 이상이어야 한다.
        if self.weight_decay < 0:
            raise ValueError("Weight Decay는 0 이상이어야 합니다.")

        if self.seed < 0 or self.seed + self.max_steps > 2**64:
            raise ValueError("epoch별 시드는 uint64 범위 안에 있어야 합니다.")

        if self.checkpoint_every <= 0:
            raise ValueError("체크포인트 저장 간격은 0보다 커야 합니다.")
