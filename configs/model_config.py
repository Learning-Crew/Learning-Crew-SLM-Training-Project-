from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    변수 의미
    vocab_size	모델이 사용하는 토큰 종류 수
    hidden_size 토큰을 표현하는 벡터의 크기
    num_layers	Transformer 블록 개수
    num_heads	Attention Head 개수
    """
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int

    def __post_init__(self):
        # 모델 설정 값 오류 잡기 

        # 모든 설정값은 양수여야 한다.
        if any(value <= 0 for value in (
            self.vocab_size, 
            self.hidden_size,
            self.num_layers, 
            self.num_heads
        )):
            raise ValueError("모델 설정값은 모두 0보다 커야 합니다.")
        
        # 각 Attention Head의 차원은 동일
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size는 num_heads로 나누어떨어져야 합니다.")