# Team 3 - SLM Training Project

## 1. 모델 개요

본 프로젝트에서는 Decoder-only Transformer 기반의 약 300M 파라미터 언어 모델을 직접 구현하고, FineWeb-Edu 데이터셋을 이용한 Pretraining과 객관식 문제에 대한 Fine-tuning을 수행하였습니다.

| 항목 | 설정 |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters | 299,125,760 (약 300M) |
| Vocabulary Size | 32,768 |
| Hidden Size | 1,024 |
| Transformer Layers | 21 |
| Attention Heads | 16 |
| Intermediate Size | 4,096 |
| Max Sequence Length | 1,024 |
| Tokenizer | Byte-level BPE |
| Position Embedding | Learned Position Embedding |
| Attention | Flash Attention 2 (PyTorch SDPA) |

## 2. 실행 환경

- Python 3.12.3
- PyTorch 2.15.0+cu132
- CUDA 13.2
- GPU: NVIDIA RTX PRO 6000 Blackwell

필요한 라이브러리는 requirements.txt에 정리되어 있습니다.

```bash
pip install -r requirements.txt
```

## 3. 폴더 구조

```text
Learning-Crew/
├── checkpoints/
│   ├── pretrained_model.pt
│   └── finetuned_model.pt
├── configs/
├── data/
├── evaluation/
├── model/
├── tokenizer/
├── tokenizer_32k/
│   ├── vocab.json
│   └── merges.json
├── training/
├── requirements.txt
└── README.md
```

- model: 모델 아키텍처 구현 코드
- configs: 모델 및 학습 설정
- tokenizer: BPE 토크나이저 구현 코드
- tokenizer_32k: 학습된 Vocabulary 및 Merge 정보
- checkpoints: Pretrained 및 Finetuned 모델
- data: 데이터 전처리 코드
- training: 모델 학습 코드
- evaluation: 모델 평가 코드

## 4. 토크나이저 로드

/home/team3/models/Learning-Crew 디렉터리를 기준으로 실행합니다.

```python
from configs.tokenizer_config import (
    TokenizerConfig,
    PretrainDatasetConfig,
)
from tokenizer.bpe_tokenizer import BPETokenizer

tokenizer = BPETokenizer(
    TokenizerConfig(
        vocab_size=32768,
        vocab_path="tokenizer_32k/vocab.json",
        merges_path="tokenizer_32k/merges.json",
    ),
    PretrainDatasetConfig(),
)
```

토크나이저의 encode() 함수는 정수 토큰 ID 리스트를 반환합니다.

```python
token_ids = tokenizer.encode("Hello, world!")
text = tokenizer.decode(token_ids)

print(token_ids)
print(text)
```

## 5. 모델 로드

Pretrained 모델과 Finetuned 모델은 동일한 모델 구조를 사용하며, 체크포인트에 저장된 model_config와 model을 이용하여 복원할 수 있습니다.

```python
import torch

from configs.model_config import ModelConfig
from model.gpt import GPTModel

device = torch.device("cuda")


def load_model(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    config = ModelConfig(**checkpoint["model_config"])
    model = GPTModel(config)

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    return model.to(device).eval()
```

### Pretrained Model

```python
model = load_model(
    "checkpoints/pretrained_model.pt"
)
```

### Finetuned Model

```python
model = load_model(
    "checkpoints/finetuned_model.pt"
)
```

객관식 문제 평가 시에는 Finetuned Model을 사용합니다.

## 6. 객관식 문제 추론 예제

교수님께서 제공하신 입력 형식에 맞춰 질문과 선택지를 구성하고, 마지막 토큰 위치의 logits를 이용하여 정답을 예측합니다.

```python
ANSWER_OPTIONS = ["A", "B", "C", "D"]


def predict(model, tokenizer, question, choices):
    choices_str = " ".join(
        f"({ANSWER_OPTIONS[i]}) {choice}"
        for i, choice in enumerate(choices)
    )

    context = f"{question.strip()}\n{choices_str}\nAnswer: "

    input_ids = [tokenizer.bos_id] + tokenizer.encode(context)

    input_ids = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device,
    )

    answer_ids = [
        tokenizer.encode(option)[0]
        for option in ANSWER_OPTIONS
    ]

    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            logits = model(input_ids)

        answer_logits = logits[0, -1, answer_ids]
        prediction = answer_logits.argmax().item()

    return ANSWER_OPTIONS[prediction]
```

사용 예시:

```python
question = "What is the capital of France?"

choices = [
    "London",
    "Paris",
    "Berlin",
    "Rome",
]

prediction = predict(
    model,
    tokenizer,
    question,
    choices,
)

print("Prediction:", prediction)
```

모델의 최대 입력 길이는 BOS 토큰을 포함하여 1,024 토큰입니다.

## 7. 최종 학습 결과

### Pretraining

| 항목 | 결과 |
|---|---|
| Dataset | FineWeb-Edu |
| Epochs | 1.5 |
| Validation Loss | 3.040622 |
| Validation Perplexity | 20.918253 |

### Fine-tuning

| 항목 | 결과 |
|---|---|
| Epochs | 3 |
| Learning Rate | 1e-5 |
| Validation Samples | 4,992 |
| Correct Predictions | 3,120 |
| Validation Loss | 0.978120 |
| Validation Accuracy | 62.50% |
| Full Vocabulary Accuracy | 62.50% |
| Invalid Predictions | 0 |

제출된 Finetuned Model은 Validation Accuracy가 가장 높았던 체크포인트입니다.