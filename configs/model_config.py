from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    변수 의미
    vocab_size	모델이 사용하는 토큰 종류 수
    hidden_size 토큰을 표현하는 벡터의 크기
    num_layers	Transformer 블록 개수
    num_heads	Attention Head 개수
    max_seq_len       모델이 처리할 수 있는 최대 토큰 길이
    intermediate_size  Feed Forward Network의 중간 차원
    norm_eps    정규화 과정에서 수치적 안정성을 위한 값
    attention_backend  Attention 계산 방식 (naive 또는 flash_attention_2) -> navie로 하면 기존의 코드로 동작합니다!
    """

    vocab_size: int = 32768
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    max_seq_len: int = 1024
    intermediate_size: int = 3072
    norm_eps: float = 1e-5
    attention_backend: str = "naive"  #GPU 학습할 때만 flash_attention_2로 바꿔서 사용!

    def __post_init__(self):
        # 모델 설정값 유효성 검사

        # 양수여야 하는 설정값
        if any(value <= 0 for value in (
            self.vocab_size, 
            self.hidden_size,
            self.num_layers, 
            self.num_heads,
            self.max_seq_len,
            self.intermediate_size,
            self.norm_eps,
        )):
            raise ValueError("모델 설정값은 모두 0보다 커야 합니다.")
        
        # 각 Attention Head의 차원은 동일
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size는 num_heads로 나누어떨어져야 합니다.")

        #오타가 있으면 naive로 돌지 않도록 바로 에러띄우게 했습니다!
        if self.attention_backend not in ("naive", "flash_attention_2"):
            raise ValueError(
                "attention_backend는 naive, flash_attention_2 중 하나여야 합니다!! 모드 설정 잘못 하셨어요!"
            )

        