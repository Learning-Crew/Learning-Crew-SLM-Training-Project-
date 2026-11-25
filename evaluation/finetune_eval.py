
import torch
import torch.nn.functional as F
from tqdm import tqdm


def extract_last_logits(logits, lengths):
    """각 샘플의 마지막 유효 위치 logits 추출."""

    batch_indices = torch.arange(
        logits.size(0),
        device=logits.device,
    )

    return logits[
        batch_indices,
        lengths.to(logits.device) - 1,
        :,
    ]


def extract_choice_logits(
    logits,
    lengths,
    answer_ids,
):
    """마지막 유효 위치에서 A/B/C/D logits 추출."""

    last_logits = extract_last_logits(
        logits,
        lengths,
    )

    answer_ids = torch.as_tensor(
        answer_ids,
        dtype=torch.long,
        device=logits.device,
    )

    return last_logits.index_select(
        dim=-1,
        index=answer_ids,
    )


@torch.inference_mode()
def evaluate_finetune(
    model,
    dataloader,
    answer_ids,
    device,
    use_bf16=True,
):
    """Full-Vocab CE 및 A/B/C/D 기준 정확도 평가."""

    was_training = model.training
    model.eval()

    answer_ids = torch.as_tensor(
        answer_ids,
        dtype=torch.long,
        device=device,
    )

    total_loss = 0.0
    total_correct = 0
    total_full_vocab_correct = 0
    total_samples = 0
    invalid_predictions = 0

    confusion = torch.zeros(
        (4, 4),
        dtype=torch.long,
    )

    try:
        for batch in tqdm(
            dataloader,
            desc="Fine-tune Validation",
        ):
            input_ids = batch["input_ids"].to(device)
            label_ids = batch["label_ids"].to(device)
            lengths = batch["lengths"].to(device)
            answers = batch["answers"].to(device)

            batch_indices = torch.arange(
                input_ids.size(0),
                device=device,
            )

            target_ids = label_ids[
                batch_indices,
                lengths - 1,
            ]

            with torch.autocast(
                device_type=torch.device(device).type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                logits = model(input_ids)

                last_logits = extract_last_logits(
                    logits,
                    lengths,
                )

            # Full-Vocabulary CrossEntropy
            loss = F.cross_entropy(
                last_logits.float(),
                target_ids,
                reduction="sum",
            )

            # A/B/C/D 기준 예측 (주 평가 지표)
            choice_logits = last_logits.index_select(
                dim=-1,
                index=answer_ids,
            )

            predictions = choice_logits.argmax(
                dim=-1
            )

            # 전체 Vocabulary 예측 (참고 지표)
            full_vocab_predictions = last_logits.argmax(
                dim=-1
            )

            total_loss += loss.item()

            total_correct += (
                predictions == answers
            ).sum().item()

            total_full_vocab_correct += (
                full_vocab_predictions == target_ids
            ).sum().item()

            total_samples += answers.size(0)

            invalid_predictions += (
                ~torch.isin(
                    full_vocab_predictions,
                    answer_ids,
                )
            ).sum().item()

            # A/B/C/D Confusion Matrix
            indices = (
                answers * 4 + predictions
            ).cpu()

            confusion += torch.bincount(
                indices,
                minlength=16,
            ).reshape(4, 4)

        if total_samples == 0:
            raise ValueError(
                "Empty validation dataset."
            )

        class_total = confusion.sum(dim=1)
        class_correct = confusion.diag()

        class_accuracy = (
            class_correct.float()
            / class_total.clamp(min=1)
        )

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
            "correct": total_correct,
            "total": total_samples,

            "full_vocab_accuracy": (
                total_full_vocab_correct / total_samples
            ),
            "full_vocab_correct": total_full_vocab_correct,
            "invalid_predictions": invalid_predictions,

            "class_accuracy": {
                label: class_accuracy[i].item()
                for i, label in enumerate("ABCD")
            },

            "prediction_counts": {
                label: confusion[:, i].sum().item()
                for i, label in enumerate("ABCD")
            },

            "confusion_matrix": confusion.tolist(),
        }

    finally:
        model.train(was_training)
