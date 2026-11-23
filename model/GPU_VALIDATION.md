# GPU 검증 및 PR #7 검토 — 이정원

## 범위와 상태

- 검토 기준: PR #7, `91c9082ba3dcc8983130ae641b11e7dc11ecc9b2`.
- 기존 모델/Trainer/팀원 Attention 구현은 변경하지 않고 독립 벤치마크를 추가한다.
- 실제 GPU 성능, 실제 데이터 loss, 언어 성능 향상은 아직 측정하지 않았다.
- 로컬 검증: Python 3.12.14 / PyTorch 2.15.0+cpu. PR #7의 Attention 테스트 8개 통과, CUDA 전용 2개 skip. 전체 저장소 테스트를 실행한 것은 아니다.
- 새 벤치마크 CPU smoke: B=1, T=8, H=32, L=1, heads=4, vocab=64, 준비 1회·측정 2회와 전후 유한성 검사 통과. 이 결과는 GPU 검증을 대체하지 않는다.

## PR 검토 결과

1. 기존 파라미터 이름을 유지하므로 naive/Flash 모델 사이 state_dict 공유가 가능하다.
2. PR에는 출력/gradient 비교와 causal 테스트가 이미 있다. GPU에서 실행해야 검증 완료다.
3. Flash 경로는 FP32 Q/K/V를 BF16으로 변환하고 Flash 커널만 허용한다. GPU 지원 여부에 따라 실패할 수 있으며 CUDA 존재만으로 지원을 보장하지 않는다.
4. main Trainer에는 mixed_precision 설정을 실제 적용하는 autocast가 보이지 않는다. Attention 내부 변환은 전체 모델 AMP 적용과 다르다. Trainer 담당자와 연동해야 한다.
5. gradient_accumulation_steps, gradient_checkpointing도 설정 존재만으로 동작한다고 판단하면 안 된다. 담당자의 구현 상태를 먼저 확인한다.
6. PR의 gradient 비교 절대오차 0.02는 작은 gradient에서 느슨할 수 있다. GPU 실측 시 gradient 크기와 오차를 함께 확인한다.

## 실행 순서

저장소 루트에서 실행한다. PR #7의 attention_backend 지원 코드가 필요하다.

```bash
python -m unittest discover -s tests -v
python -m benchmarks.gpu_model --device cpu --precision fp32 --backend naive --steps 2 --warmup 1 --hidden-size 32 --heads 4 --layers 1 --seq-len 8 --vocab-size 64 --source-revision "$(git rev-parse HEAD)" --output /tmp/slm-cpu-smoke.json
```

서버에서는 GPU를 다른 실험이 사용하지 않는 시간에 아래 두 명령을 순서대로 실행한다.

```bash
python -m benchmarks.gpu_model --backend naive --precision bf16 --source-revision "$(git rev-parse HEAD)" --output /tmp/slm-naive-bf16.json
python -m benchmarks.gpu_model --backend flash_attention_2 --precision bf16 --source-revision "$(git rev-parse HEAD)" --output /tmp/slm-flash-bf16.json
```

BF16 미지원이면 두 실행 모두 fp16으로 바꾼다. Flash 커널 미지원 오류는 그대로 기록하고 다른 backend 실행 결과를 Flash 결과로 표시하지 않는다. 결과 파일이 이미 있으면 새 이름을 사용한다.

작은 모델 검증 뒤 실제 모델 설정으로 두 명령 모두 다음 인자를 추가한다.

```text
--vocab-size 32768 --hidden-size 768 --layers 12 --heads 12 --seq-len 1024 --batch-size 1
```

실제 확정 vocab/model 설정이 다르면 그 값으로 맞춘다. 이후 배치 크기만 1, 2, 4 순으로 늘리며 측정한다. OOM이면 해당 조건을 실패로 기록하고 프로세스를 새로 실행한다. 각 조건을 3회 측정하고 중앙값과 범위를 기록한다.

## 측정 방법과 해석

- 같은 seed, 모델 크기, 배치, 정밀도, 입력을 사용하고 backend만 변경한다.
- 준비 단계 뒤 CUDA synchronize로 시간 경계를 맞춘다. forward, loss, backward, clipping, AdamW를 포함한다.
- 초기 검증 및 warmup 후 peak memory 통계를 초기화한다. allocated/reserved를 모두 기록한다.
- 측정 전후 loss, 모든 parameter gradient와 마지막 parameter의 유한성을 검사한다. 매 측정 step의 전체 gradient 검사는 하지 않는다.
- FP16은 GradScaler를 사용한다. 측정 중 scale 감소를 발견하면 결과 저장 없이 실패시킨다.
- 고정 합성 토큰을 사용하므로 tokenizer, 데이터 로딩, checkpoint 저장, validation 시간은 포함하지 않는다.
- 현재 Trainer를 호출하지 않는다. 여기서의 AMP 속도를 현재 Trainer의 end-to-end 속도로 보고하면 안 된다.
- 이 loss 변화는 언어 모델의 성능 향상 근거가 아니다. 실제 데이터 평가는 별도로 수행한다.
- CPU 결과는 실행 경로 확인에만 사용하고 GPU 성능 비교에 사용하지 않는다.

## 팀 공유용 결과 항목

commit / GPU / PyTorch·CUDA / backend / dtype / B·T·모델 크기 / tokens/sec / peak allocated·reserved / 유한성 검사 / 오류.

실측값이 생기면 결과 JSON을 근거로 PROGRESS.md를 갱신한다. 팀장에게 학습 가능한 배치 범위와 처리량을 전달하고, 메모리가 부족할 때만 모델 checkpointing 추가를 논의한다.
