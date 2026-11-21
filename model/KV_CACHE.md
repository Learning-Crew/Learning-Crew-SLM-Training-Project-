# KV cache 추론 기능

`forward(input_ids)`의 `(B,T,V)` logits 반환과 학습 경로는 그대로 유지한다.
새 `forward_cached(input_ids, past_key_values=None)`는 `(logits, cache)`를 반환한다.
학습 파라미터를 추가하지 않아 기존 state_dict를 strict=True로 불러올 수 있다.
캐시는 실행 중 별도로 보관하며 체크포인트에 저장하지 않는다.

## 사용 예시

기존 체크포인트에서 복원한 모델과 토크나이저로 만든 prompt_ids를 사용한다.
아래는 토큰 생성 예시이며 EOS/샘플링/문자열 변환은 평가 코드에서 처리한다.

```python
import torch

model.eval()
# prompt_ids: padding 없는 torch.long, (B,T), 모델과 같은 device
logits, cache = model.forward_cached(prompt_ids)
output_ids = prompt_ids
# 위치 임베딩의 최대 길이를 넘기지 않는다.
steps = min(20, model.max_seq_len - prompt_ids.shape[1])
for step in range(steps):
    next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
    output_ids = torch.cat((output_ids, next_id), dim=1)
    if step + 1 < steps:
        logits, cache = model.forward_cached(next_id, cache)
```

GPU BF16/FP16에서는 prefill과 모든 decode 호출을 동일한 autocast 문맥 안에 둔다.
CUDA Flash Attention은 기존 backend 설정을 따른다. 캐시가 있는 단일 토큰은
모든 과거 key를 볼 수 있으므로 is_causal=False를 사용한다. prefill은 기존
causal attention을 사용한다. 별도의 새로운 가중치나 외부 모델은 사용하지 않는다.

## 범위 및 제한

- eval 전용이며 내부 no_grad 적용. 기존 forward로 학습한다.
- 첫 호출은 문장 전체(prefill), 이후 호출은 새 토큰 한 개만 입력한다.
- 캐시는 층별 (key,value) tuple, 각 tensor는 (B,heads,T,head_dim)이다.
- 동일 모델 가중치, 배치 순서, device, 정밀도 설정에서만 캐시를 재사용한다.
- 새 문장이나 가중치 변경 후에는 cache=None으로 다시 시작한다.
- padding, 서로 다른 길이의 배치, beam reorder, chunk decode는 지원하지 않는다.
- 최대 길이에 도달하면 중단한다. learned position 때문에 캐시를 임의로
  잘라 슬라이딩 윈도우처럼 사용하는 것은 지원하지 않는다.
- 과거 KV를 저장하므로 메모리를 더 사용한다. 사전학습 속도 개선 기능은 아니다.
- 현재 동적 torch.cat 캐시이며 GPU 생성 속도 향상률은 아직 측정하지 않았다.

## 검증

```bash
python -m unittest discover -s tests -p 'test_kv_cache.py' -v
python -m unittest discover -s tests -v
```

전체 prefix 계산과 캐시 계산의 logits, greedy 생성 결과, 캐시 길이와 입력
검증, 파라미터 불변성, 기존 학습 gradient 및 gradient checkpointing을 검사한다.
CPU SDPA 테스트는 mask 정합성 검사이며 실제 CUDA Flash Attention 검증과 다르다.
CUDA가 없으면 GPU 전용 테스트는 skip된다.

검증 기록 (2026-11-21): Python 3.12 / PyTorch 2.15.0+cpu, CPU에서
전체 31개 테스트 중 28개 통과, CUDA 전용 3개 skip. 새 캐시 테스트는
6개 중 CPU 5개 통과, CUDA 1개 skip. 실제 GPU 속도/메모리 측정은 미실행.

변경 전 main(e2526f5)의 작은 모델 state_dict를 strict=True로 불러온 뒤,
기존 forward logits와 모든 파라미터 gradient가 bit 단위로 동일함도 확인했다.
