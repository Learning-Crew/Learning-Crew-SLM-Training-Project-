import torch
import torch.nn.functional as F

# 토큰 정확도 계산
def token_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> float:
    
    predictions = torch.argmax(logits, dim=-1)

    valid_mask = (targets != -100)

    correct = (predictions == targets) & valid_mask
    correct_count = correct.sum()

    total_count = valid_mask.sum().item()
    if total_count == 0:
        return 0.0

    accuracy = correct_count.item() / total_count

    return accuracy


# PPL 지표 (loss : 평균 CEL)
def perplexity(loss: float) -> float:
    ppl = torch.exp(torch.tensor(loss))

    return ppl.item()


# Cross_Entropy_loss 계산
def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:

    # [B,T,V] -> [B,V,T] 변환 (F.cross_entropy는 [B,V,T] 형태로 사용되기 때문)
    logits = logits.transpose(1, 2)
    loss = F.cross_entropy(logits, targets, ignore_index=-100)

    return loss


if __name__ == "__main__":
    logits = torch.tensor([
        [
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5],
            [0.9, 0.05, 0.05],
            [0.1, 0.2, 0.7],
        ]
    ])

    targets = torch.tensor([[1, 2, -100, 2]])

    accuracy = token_accuracy(logits, targets)
    print(f"Token Accuracy: {accuracy:.2%}")

    loss = cross_entropy_loss(logits, targets)
    print(f"Cross Entropy Loss: {loss.item():.4f}")

    ppl = perplexity(loss.item())
    print(f"Perplexity: {ppl:.4f}")