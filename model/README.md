# GPT 모델 사용

저장소 루트에서 실행합니다. `forward()`는 loss나 확률이 아닌 logits 텐서만 반환합니다.

```python
import torch
from configs.model_config import ModelConfig
from model.gpt import GPTModel

config = ModelConfig(vocab_size=100, hidden_size=32, num_layers=2, num_heads=4)
model = GPTModel(config, max_seq_len=128)
input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
logits = model(input_ids)
print(logits.shape)  # torch.Size([2, 3, 100])
```

실제 `vocab_size`는 토크나이저 담당자가 공유한 값으로 설정합니다.
`max_seq_len`은 데이터의 시퀀스 길이 이상으로 팀과 합의합니다.
위 숫자는 동작 확인용이며 최종 모델 규모가 아닙니다.

## 학습 담당자 연결

데이터의 `input_ids`와 `targets`는 이미 한 칸 이동되어 있습니다.
모델 또는 학습 단계에서 추가로 shift하지 않습니다.

```python
loss = torch.nn.functional.cross_entropy(
    logits.reshape(-1, config.vocab_size), targets.reshape(-1)
)
```

`targets`는 입력과 같은 `(B, T)` 크기의 LongTensor입니다.
GPU 사용 시 모델, 입력, targets를 같은 장치로 이동합니다.

## 체크포인트 연결

학습 담당자의 체크포인트에 `model.state_dict()`와
`model.get_model_config()`를 저장합니다. 후자는 공통 설정 네 값과
`max_seq_len`을 포함합니다. Optimizer와 Step 저장은 학습 담당 영역입니다.

```python
meta = checkpoint["model_config"]
model = GPTModel(ModelConfig(**meta["config"]), max_seq_len=meta["max_seq_len"])
model.load_state_dict(checkpoint["model_state_dict"])
```

키 이름은 위 예시를 기준으로 학습 담당자와 맞춥니다.
공통 ModelConfig와 INTERFACES.md는 변경하지 않았습니다.

## 범위와 검증

현재 구조: 학습 가능한 절대 위치 Embedding, causal multi-head attention,
Pre-LN 블록, GELU FFN, 최종 LayerNorm, 입력/출력 가중치 공유.
패딩 없는 연속 토큰 시퀀스를 지원합니다. Padding mask, KV cache,
dropout, 텍스트 생성 함수는 포함하지 않습니다.
Attention은 명시적 행렬 계산이므로 긴 시퀀스에서 메모리가 제곱으로 증가합니다.
125M~150M 규모의 실제 GPU 학습 속도와 메모리는 별도 측정이 필요합니다.

```bash
python -m unittest discover -s tests -v
```

작은 데이터 학습 검증은 연산 연결을 확인하기 위한 것이며 실제 언어 성능의 증거가 아닙니다.
