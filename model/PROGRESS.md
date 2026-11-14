# 모델 파트 진행 상황 — 이정원

최종 정리: 2026-11-15 (KST)  
점검 기준: main 커밋 `3a9d0b94ce584e54a82f6c61d261811449ed13f6`

## 현재 상태

**기본 GPT 모델 구현과 CPU 검증 완료, main 반영 완료. 실제 데이터 학습과 GPU 검증은 진행 전.**

담당 범위는 Model Architecture이며, 모델 본체와 모델 검증 코드를 구현했습니다.
[PR #5](https://github.com/Learning-Crew/Learning-Crew-SLM-Training-Project-/pull/5)는 병합 완료됐습니다.
기존 구현 커밋 이력을 유지한 merge 방식으로 반영됐으며, 팀장 검토 완료를 의미하지는 않습니다.

## 완료한 작업

| 구성 | 구현 내용 | 코드 |
|---|---|---|
| Embedding | 토큰 벡터와 학습 가능한 절대 위치 벡터를 합산 | [embeddings.py](embeddings.py) |
| Causal Self-Attention | Q/K/V, Multi-head Attention, 미래 토큰 차단, 출력 투영 | [attention.py](attention.py) |
| Transformer Block | Pre-LayerNorm, GELU FFN, 잔차 연결, 잔차 출력 초기화 | [transformer_block.py](transformer_block.py) |
| GPT 모델 | Embedding과 블록 연결, 최종 LayerNorm, 입력·출력 가중치 공유 | [gpt.py](gpt.py) |
| 공통 설정 연결 | max_seq_len, intermediate_size, norm_eps 등 ModelConfig 반영 | [gpt.py](gpt.py), [transformer_block.py](transformer_block.py) |
| 평가 모듈 연결 | 실제 GPT 모델을 evaluation.evaluate에 전달해 배치별 결과 일치 검증 | [test_integration.py](../tests/test_integration.py) |

공통 Config, INTERFACES.md 및 다른 담당자의 구현은 이 PR에서 수정하지 않았습니다.

## 검증 기록

최신 main과 모델 변경을 합친 로컬 트리에서 **CPU 테스트 21개 통과**를 확인했습니다.
GPU에서의 결과는 아닙니다.

- 입력·출력 크기 및 잘못된 입력 처리
- 미래 토큰 변경 시 앞쪽 출력 불변, 미래 위치로의 gradient 차단
- 직접 구현한 Attention과 PyTorch SDPA 기준 연산의 출력 비교
- Embedding부터 전체 모델까지 유한한 gradient 전달
- 여러 Transformer Block의 연결 및 잔차 경로
- 가중치와 설정 저장·복원 후 출력 일치
- 작은 반복 데이터의 next-token 학습
- 공통 Config 적용 및 Evaluation 배치 크기 1/2의 결과 일치

최신 통합 검증에서 작은 반복 데이터의 Loss는 40회 업데이트 동안
**2.8209 → 0.0318**로 감소했습니다. 이는 연산과 학습 연결의 확인용 결과이며,
실제 언어 성능이나 벤치마크 점수가 아닙니다. 앞선 작은 모델 검증과 설정이
달라 수치를 성능 개선 비교로 사용하지 않습니다.

재실행:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
```

## 팀원 연결 방법

```python
from configs.model_config import ModelConfig
from model.gpt import GPTModel

config = ModelConfig()
model = GPTModel(config)
logits = model(input_ids)
```

| 항목 | 규칙 |
|---|---|
| input_ids | torch.long, (B, T), 토큰 ID는 0 이상 vocab_size 미만 |
| logits | (B, T, V)의 텐서만 반환. softmax와 Loss 계산은 포함하지 않음 |
| targets | Dataset에서 이미 한 칸 이동된 (B, T) 정답 사용. 추가 shift 금지 |
| 문장 길이 | T는 config.max_seq_len 이하 |
| 실행 장치 | 모델, input_ids, targets를 동일한 장치로 이동 |
| 체크포인트 | model.state_dict()와 model.get_model_config() 저장. Optimizer·Step은 학습 담당 코드에서 저장 |

체크포인트 키 이름과 생성·복원 호출은 학습 담당자와 함께 맞춰야 합니다.
자세한 예시는 [모델 사용 설명](README.md)을 참고합니다.

## 통합 시 확인할 사항

- [ ] 토크나이저의 실제 어휘 수와 특수 토큰 ID가 ModelConfig와 일치하는지 확인
- [ ] 실제 데이터의 시퀀스 길이와 입력·정답 shift 확인
- [ ] 학습 담당 코드에서 한 배치의 Loss 계산, 역전파, Optimizer 업데이트 확인
- [ ] 학습 체크포인트 저장·재개와 평가 코드 연결 확인
- [ ] 팀장 통합 과정에서 필요한 모델 인터페이스 조정

Evaluation 연결 테스트는 작은 인공 입력을 사용했습니다.
실제 데이터 로딩부터 학습까지 전체 파이프라인을 검증했다는 의미는 아닙니다.

## 현재 한계 및 미완료

- 실제 Pretraining, Finetuning, 최종 벤치마크는 아직 수행하지 않았습니다.
- GPU 속도, 최대 메모리 및 최종 모델 규모는 검증하지 않았습니다.
- 패딩 없는 연속 토큰 시퀀스를 기준으로 합니다. Padding mask와 KV cache는 미구현입니다.
- Attention은 명시적 행렬 계산이므로 긴 시퀀스에서 메모리 부담이 커질 수 있습니다.
- SDPA는 테스트의 비교 기준으로만 사용했고, 모델 실행 경로에는 아직 적용하지 않았습니다.
- 텍스트 생성 및 실제 학습 루프는 해당 담당자와 연결할 영역입니다.

## 다음 작업

1. 팀 전체 기본 파이프라인 통합을 우선 지원합니다.
2. 기본 연결 확인 후 기존 Attention과 SDPA를 선택·비교할 수 있도록 개선합니다.
3. 출력·gradient 동등성을 검증하고 GPU 속도·메모리 측정 도구를 준비합니다.
4. GPU 확보 후 동일 조건으로 측정하고 설정·환경·결과를 기록합니다.

## 기록 원칙

완료 여부와 실험 수치는 실제 확인된 범위만 기록합니다.
기능 개발 및 검증에 AI 보조 도구를 사용했으며, 수업의 AI 활용·출처 표기 규정은 별도로 확인해야 합니다.
커밋 수와 파일 줄 수만으로 과제의 개인 기여 인정 여부를 확정하지 않습니다.
