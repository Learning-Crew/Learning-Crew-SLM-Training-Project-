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
- 기본 tuple 캐시는 동적 torch.cat 방식이다. 아래 사전 할당 캐시도 선택할 수 있다.
- 두 방식 모두 GPU 생성 속도 향상률은 아직 측정하지 않았다.

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


## 사전 할당 캐시 (2026-11-23)

전체 과거 KV를 매번 복사하는 대신 각 층의 공간을 한 번 확보하고 새 위치만
기록한다. 실제 attention에는 기록된 구간만 전달한다. 이 구현은 shape가
증가하는 view를 사용하므로 torch.compile/CUDA Graph 최적화를 보장하지 않는다.
학습 정확도 개선이나 모든 하드웨어에서의 최적 속도를 주장하지 않는다.

```python
model.eval()
# prompt와 생성할 토큰이 들어갈 만큼만 잡아 메모리 낭비를 줄인다.
capacity = min(model.max_seq_len, prompt_ids.shape[1] + 20)
cache = model.create_static_cache(capacity=capacity)
logits, cache = model.forward_cached(prompt_ids, cache, last_logits_only=True)
# logits: (B, 1, V). 전체 prompt의 vocabulary projection을 생략한다.
next_id = logits[:, -1].argmax(-1, keepdim=True)
if cache.length < capacity:
    logits, cache = model.forward_cached(next_id, cache, last_logits_only=True)
# 다른 prompt: 동일 batch/device/dtype이면 storage를 재사용할 수 있다.
cache.reset()
```

- cache는 제자리에서 수정된다. 요청마다 별도로 만들고 동시 실행에 공유하지 않는다.
- reset은 길이만 0으로 되돌린다. 이전 내용은 attention에서 보이지 않는다.
- batch/device/dtype 변경 시 새 cache를 만든다. 가중치 변경 후에도 새로 시작한다.
- 일부 층만 기록한 상태에서 예외가 발생하면 모든 층의 길이를 0으로 초기화한다.
- `last_logits_only=True`는 생성용이다. 전체 토큰 loss 평가에는 기본값 False를 쓴다.
- KV 메모리는 `2 × layers × batch × capacity × hidden × dtype bytes`이다.
  예: 21층, hidden 1024, batch 1, capacity 1024, BF16이면 84 MiB.
  이는 KV만의 용량이고 모델 가중치/활성값/작업 공간은 별도이다.
- cache.allocated_bytes로 실제 할당한 KV tensor 크기를 확인할 수 있다.

### 비교 실행

```bash
OMP_NUM_THREADS=1 python -m benchmarks.kv_cache --device cpu
python -m benchmarks.kv_cache --device cuda --dtype bfloat16 --backend flash_attention_2 --hidden 1024 --layers 21 --heads 16 --vocab 32768 --prompt 512 --decode 128 --output kv_gpu.json
```

GPU 명령은 팀의 본 학습과 겹치지 않는 시간에 실행한다. synthetic token ID와
무작위 초기 가중치를 쓰는 속도 비교이며, 학습 완료 모델의 품질 평가가 아니다.
각 방식은 같은 가중치/입력으로 prefill과 고정 토큰 decode를 수행한다.
동적/정적 모두 last_logits_only를 켠다. full 방식은 기존 forward 그대로 사용한다.
워밍업 및 최종 logits 일치 검사 후 순서를 번갈아 가며 5회 측정한다.
GPU 시간 측정은 synchronize를 사용하며 peak allocated memory도 기록한다.

검증 환경: Python 3.12.14 / PyTorch 2.5.1+cpu. 전체 36개 중 32개 통과,
CUDA 4개 skip. 사전 할당 캐시의 전체-prefix 동등성, 주소 불변/torch.cat 미사용,
용량 초과, reset 재사용, 오류 후 복구, 정밀도 변경 거부를 검증했다.

CPU 1 thread, hidden128/layers2/heads4/vocab1024, batch1/prompt64/decode32:
5회 중앙값 full 69.29ms / dynamic 20.52ms / static 19.11ms.
작은 CPU 사례이므로 GPU 성능이나 일반적인 우위를 보장하지 않는다.
원시 측정은 benchmarks/results/kv_cpu_20261123.json에 있다.

참고: https://huggingface.co/docs/transformers/kv_cache
사전 할당 전략을 참고했으며 transformers 라이브러리 의존성은 추가하지 않았다.
