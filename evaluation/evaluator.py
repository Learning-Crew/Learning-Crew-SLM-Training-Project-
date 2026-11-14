import torch
from torch.utils.data import DataLoader
from evaluation.metrics import token_accuracy, cross_entropy_loss, perplexity

def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> dict: 
    
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids, targets = batch
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            logits = model(input_ids)
           
            # 유효 토큰 수 계산
            valid_mask = (targets != -100)
            num_tokens = valid_mask.sum().item()

            # 유효 토큰이 없는 배치는 건너뛰기
            if num_tokens == 0:
                continue

            # 현재 배치의 Loss와 Accuracy 계산
            loss = cross_entropy_loss(logits, targets)
            accuracy = token_accuracy(logits, targets)

            # 전체 Loss와 유효 토큰 수 누적
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            # 배치 Accuracy를 이용해 정답 토큰 수 누적
            total_correct += accuracy * num_tokens

        if total_tokens == 0:
            return {
                "loss": 0.0,
                "accuracy": 0.0,
                "perplexity": 0.0,
            }

        avg_loss = total_loss / total_tokens
        avg_accuracy = total_correct / total_tokens
        ppl = perplexity(avg_loss)

        return {
            "loss": avg_loss,
            "accuracy": avg_accuracy,
            "perplexity": ppl,
        }