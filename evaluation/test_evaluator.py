import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluation.evaluator import evaluate


# 실제 모델 대신 사용할 테스트용 모델
class DummyModel(nn.Module):
    def __init__(self, vocab_size=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 16)
        self.fc = nn.Linear(16, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        logits = self.fc(x)
        return logits

if __name__ == "__main__":

    # 실행 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 테스트용 입력 토큰과 정답 토큰
    dataset = [
        {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "labels": torch.tensor([2, 3, 4, 5])
        },
        {
            "input_ids": torch.tensor([3, 4, 5, 6]),
            "labels": torch.tensor([4, 5, 6, 7])
        }
    ]

    # 배치 크기를 2로 설정
    dataloader = DataLoader(dataset, batch_size=1)

    # 동일한 모델을 사용해 배치 크기별 결과 비교
    torch.manual_seed(42)
    model = DummyModel(vocab_size=10).to(device)

    # 배치 크기 1
    loader1 = DataLoader(dataset, batch_size=1)
    result1 = evaluate(model, loader1, device)

    # 배치 크기 2
    loader2 = DataLoader(dataset, batch_size=2)
    result2 = evaluate(model, loader2, device)

    print("Batch Size 1:", result1)
    print("Batch Size 2:", result2)