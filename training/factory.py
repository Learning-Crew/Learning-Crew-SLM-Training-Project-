
from configs.model_config import ModelConfig
from configs.tokenizer_config import PretrainDatasetConfig, TokenizerConfig

from data.tokenized_dataset import TokenizedDataset

from tokenizer.bpe_tokenizer import BPETokenizer

from model.gpt import GPTModel

from itertools import islice
from datasets import load_dataset



def build_experiment(
    model_config: ModelConfig,
    dataset_config: PretrainDatasetConfig,
    tokenizer_config: TokenizerConfig,
):
    #설정값을 사용해서 GPT 모델과 BPE 토크나이저 생성
    model = GPTModel(model_config)
    tokenizer = BPETokenizer(
        config=tokenizer_config,
        dataset_config=dataset_config,
    )

    #실제 데이터 연결 전에 작은 샘플로 테스트
    #로컬 FineWeb-Edu JSONL 데이터에서 일부 문서 읽기
    train_stream = load_dataset(
    "json",
    data_files="data/train.jsonl",
    split="train",
    streaming=True
    )

    validation_stream = load_dataset(
        "json",
        data_files="data/valid.jsonl",
        split="train",
        streaming=True
    )

    #전체 데이터 대신 일부 문서만 사용
    train_documents = list(islice(train_stream, 20))
    validation_documents = list(islice(validation_stream, 5))

    #텍스트를 토큰 ID로 변환하고 입력과 정답 생성
    train_data = TokenizedDataset(
        documents=train_documents,
        tokenizer=tokenizer,
        seq_len=model_config.max_seq_len,
    )

    validation_data = TokenizedDataset(
        documents=validation_documents,
        tokenizer=tokenizer,
        seq_len=model_config.max_seq_len,
    )

    #Trainer에서 사용할 모델, 데이터, 데이터 식별자 반환
    return model, train_data, validation_data, "fineweb-edu-stream-test-v1"
