
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    변수 의미
    learning_rate                초기 학습률
    batch_size                   한 번에 처리하는 데이터 개수
    max_steps                    최대 학습 업데이트 횟수
    weight_decay                 가중치 감소 계수
    adam_betas                   AdamW의 모멘텀 계산 계수
    adam_eps                     AdamW의 수치 안정성을 위한 값
    warmup_steps                 학습률을 점진적으로 증가시키는 횟수
    min_lr                       최소 학습률
    gradient_accumulation_steps  기울기를 누적할 미니배치 횟수
    max_grad_norm                Gradient Clipping 기준값
    gradient_checkpointing       GPU 메모리 절약 기능 사용 여부
    mixed_precision              혼합 정밀도 연산 방식
    seed                         난수 시드
    checkpoint_every             체크포인트 저장 주기
    log_every                    학습 로그 출력 주기
    """

    # 기본 학습 설정
    learning_rate: float = 3e-4
    batch_size: int = 8
    max_steps: int = 10000
    weight_decay: float = 0.1

    # Optimizer 설정
    adam_betas: tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-8

    # Scheduler 설정
    warmup_steps: int = 500
    min_lr: float = 3e-5

    # Gradient 설정
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False

    # GPU 설정
    mixed_precision: str = "bf16"
    seed: int = 42

    # 체크포인트 및 로그 설정
    checkpoint_every: int = 1000
    log_every: int = 100

    def __post_init__(self):
        # 학습 설정값 유효성 검사
        
        # 양수여야 하는 설정값
        if any(value <= 0 for value in (
            self.learning_rate,
            self.batch_size,
            self.max_steps,
            self.adam_eps,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
            self.checkpoint_every,
            self.log_every,
        )):
            raise ValueError("해당 학습 설정값은 0보다 커야 합니다.")

        # 0 이상이어야 하는 설정값
        if any(value < 0 for value in (
            self.weight_decay,
            self.warmup_steps,
            self.min_lr,
        )):
            raise ValueError("해당 학습 설정값은 0 이상이어야 합니다.")

        # 설정값 간 관계 검사
        if self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps는 max_steps보다 작아야 합니다.")

        if self.min_lr > self.learning_rate:
            raise ValueError("min_lr은 learning_rate보다 클 수 없습니다.")

        if len(self.adam_betas) != 2 or not all(
            0 <= beta < 1 for beta in self.adam_betas
        ):
            raise ValueError("adam_betas는 0 이상 1 미만인 두 값이어야 합니다.")

        if self.mixed_precision not in ("no", "fp16", "bf16"):
            raise ValueError(
                "mixed_precision은 no, fp16, bf16 중 하나여야 합니다."
            )
