# SLM 공통 인터페이스

## 1. Tokenizer

- 입력: `str`
- 출력: `list[int]`
- 특수 토큰 ID와 Vocabulary 크기는 토크나이저 담당자가 공유한다.

## 2. Dataset

- `input_ids`: `torch.LongTensor`, Shape `(B, T)`
- `targets`: `torch.LongTensor`, Shape `(B, T)`
- 다음 토큰 예측을 위해 입력과 정답을 한 칸씩 이동하여 구성한다.

## 3. Model

- 입력: `input_ids`, Shape `(B, T)`
- 출력: `logits`, Shape `(B, T, V)`
- `B`: Batch Size
- `T`: Sequence Length
- `V`: Vocabulary Size
- 모델의 `forward()`는 기본적으로 Logits Tensor를 반환한다.

## 4. Training

- 모델 출력 Logits와 Targets를 사용해 Cross Entropy Loss를 계산한다.
- 학습 설정은 `configs/training_config.py`를 사용한다.
- 체크포인트에는 모델 가중치, Optimizer 상태, 학습 Step, 모델 설정을 저장한다.

## 5. Evaluation

- 입력: 모델, 검증 데이터
- 출력: 평균 Loss, Perplexity(PPL)
- 텍스트 생성 기능은 별도 함수로 구현한다.

## 6. 협업 규칙

- 모델 구조 설정은 `ModelConfig`를 기준으로 한다.
- 학습 설정은 `TrainingConfig`를 기준으로 한다.
- 공통 입출력 규칙을 변경할 때는 팀원들과 먼저 합의한다.