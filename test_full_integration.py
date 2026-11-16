
import logging
import tempfile
from pathlib import Path

import torch

from configs.model_config import ModelConfig
from configs.training_config import TrainingConfig
from configs.tokenizer_config import TokenizerConfig, PretrainDatasetConfig

from tokenizer.bpe_tokenizer import BPETokenizer
from data.tokenized_dataset import TokenizedDataset
from model.gpt import GPTModel
from training.trainer import train


#학습 로그 출력
logging.basicConfig(level=logging.INFO)

#난수 시드 고정
torch.manual_seed(42)


#실제 모델보다 작은 크기로 통합 테스트
model_config = ModelConfig(
    vocab_size=32768,
    hidden_size=64,
    num_layers=2,
    num_heads=4,
    max_seq_len=16,
    intermediate_size=256,
)

#3스텝만 학습해서 정상 작동 확인
training_config = TrainingConfig(
    batch_size=2,
    max_steps=3,
    warmup_steps=0,
    log_every=1,
    checkpoint_every=3,
)

#BPE 토크나이저 생성
tokenizer = BPETokenizer(
    config=TokenizerConfig(),
    dataset_config=PretrainDatasetConfig(),
)

#모델과 토크나이저의 vocab_size 일치 확인
assert tokenizer.vocab_size == model_config.vocab_size


#실제 데이터 대신 작은 텍스트 샘플 사용
documents = [
    {"text": "Hello world! I am learning BPE."},
    {"text": "This is a small language model training test."},
    {"text": "We are connecting the tokenizer to the dataset."},
    {"text": "GPT predicts the next token in a sequence."},
    {"text": "Deep learning models are trained with gradients."},
]

#텍스트를 토큰 ID로 변환하고 입력, 정답 생성
dataset = TokenizedDataset(
    documents=documents,
    tokenizer=tokenizer,
    seq_len=16,
)

#모델 생성
model = GPTModel(model_config)

print("학습 샘플 수:", len(dataset))
print("모델 파라미터 수:", sum(p.numel() for p in model.parameters()))


#임시 폴더에 체크포인트 저장
with tempfile.TemporaryDirectory() as checkpoint_dir:

    #실제 GPT 모델로 3스텝 학습
    final_step = train(
        model=model,
        dataset=dataset,
        model_config=model_config,
        training_config=training_config,
        device="cpu",
        dataset_id="tokenized-integration-test",
        checkpoint_dir=checkpoint_dir,
    )

    #3스텝 학습 완료 확인
    assert final_step == 3

    #체크포인트 파일 생성 확인
    assert (Path(checkpoint_dir) / "step_3.pt").is_file()


print("모델 전체 통합 테스트 성공!")
