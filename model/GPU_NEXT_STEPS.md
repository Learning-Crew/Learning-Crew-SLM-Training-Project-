# 전체 모델 수치 검증 — 별도 작업

이 작업은 feature/model-gpu-validation 브랜치에만 저장한다. 새 PR은 생성하지 않는다.

## 추가 검증

`benchmarks/compare_model.py`는 같은 가중치·입력·정밀도에서 전체 GPT의 logits, loss, 모든 학습 파라미터 gradient를 비교한다. 미래 토큰을 바꿨을 때 앞쪽 출력이 유지되는지도 검사한다.

각 항목의 최대 절대오차, 상대 L2 오차, 기준 gradient 크기를 JSON에 기록한다. 기본 판정은 상대 L2 5% 이내 또는 절대오차 1e-7 이내다. 이 값은 실험을 위한 초기 기준이며, 통과가 장기 학습 품질을 보증하지 않는다. 실패하면 허용오차부터 늘리지 말고 실패한 항목과 precision을 확인한다.

## 서버에서 순서대로 실행

먼저 팀 저장소의 본인 작업 폴더에서 아래 명령을 실행한다. 미커밋 변경이 있으면 브랜치 전환 전에 보존한다.

```bash
git fetch origin
git switch feature/model-gpu-validation
git pull --ff-only origin feature/model-gpu-validation
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

GPU가 다른 학습에 사용 중이면 측정 시간을 팀원과 조율한다. 아래 실행은 짧은 합성 입력 검사이며 전체 데이터 학습을 시작하지 않는다.

```bash
python -m benchmarks.compare_model --source-revision "$(git rev-parse HEAD)" --output /tmp/slm-model-parity-bf16.json
```

BF16 미지원이면 `--precision fp16`을 추가하고 파일명도 fp16으로 바꾼다. 이 비교에서는 gradient를 직접 검사하기 위해 GradScaler를 사용하지 않는다. FP16 underflow가 의심되면 결과와 함께 보고하고 실제 Trainer의 scaling 경로를 별도로 검증한다.

통과 후 `GPU_VALIDATION.md`의 naive/Flash 속도 측정을 순서대로 실행한다. 수치 검증 JSON과 속도 측정 JSON을 함께 보관한다. 실패 시 JSON(생성된 경우)과 오류 전문을 보관한다.

## CPU 자체 검사

```bash
python -m unittest benchmarks.test_compare_model -v
python -m benchmarks.compare_model --cpu-self-check --source-revision "$(git rev-parse HEAD)" --output /tmp/slm-model-self-check.json
```

CPU 모드는 naive를 naive와 비교하여 검증 코드 자체를 점검한다. FlashAttention 검증 결과로 해석하면 안 된다. 실제 GPU 결과는 아직 없다.

로컬 결과: PyTorch 2.15.0+cpu에서 비교 함수 테스트 3개 통과(동일 값, 작은 gradient 오류 탐지, NaN 탐지). 전체 GPT CPU 자체 검사도 통과했다. GPU 검증은 미실행이다.
