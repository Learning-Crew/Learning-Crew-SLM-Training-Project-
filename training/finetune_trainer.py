import json
import math
import time
from pathlib import Path
from dataclasses import asdict

import torch
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.checkpoint import checkpoint

from training.checkpoint import save_checkpoint
from evaluation.finetune_eval import evaluate_finetune


def forward_hidden(model, input_ids):
    hidden = model.embedding(input_ids)

    for block in model.blocks:
        if model.gradient_checkpointing and model.training:
            hidden = checkpoint(
                block, hidden, use_reentrant=False
            )
        else:
            hidden = block(hidden)

    return hidden


def forward_vocab_logits(model, input_ids, lengths):
    hidden = forward_hidden(model, input_ids)

    indices = torch.arange(
        input_ids.size(0), device=input_ids.device
    )

    last_hidden = hidden[indices, lengths - 1]
    last_hidden = model.final_norm(last_hidden)

    return F.linear(
        last_hidden,
        model.embedding.token_embedding.weight,
    )


def forward_choice_logits(model, input_ids, lengths, answer_ids):
    logits = forward_vocab_logits(model, input_ids, lengths)

    return logits.index_select(-1, answer_ids)


def forward_context_answer_loss(
    model,
    input_ids,
    lengths,
    target_ids,
    answer_ids,
    context_weight=0.1,
    label_smoothing=0.05,
):
    hidden = forward_hidden(model, input_ids)
    hidden = model.final_norm(hidden)

    weight = model.embedding.token_embedding.weight

    indices = torch.arange(
        input_ids.size(0), device=input_ids.device
    )

    # 1. Answer Loss + Label Smoothing
    last_hidden = hidden[indices, lengths - 1]
    vocab_logits = F.linear(last_hidden, weight)

    log_probs = F.log_softmax(
        vocab_logits.float(), dim=-1
    )

    hard_loss = F.nll_loss(
        log_probs, target_ids, reduction="sum"
    )

    choice_log_probs = log_probs.index_select(
        -1, answer_ids
    )

    smooth_loss = -choice_log_probs.mean(dim=-1).sum()

    answer_loss = (
        (1 - label_smoothing) * hard_loss
        + label_smoothing * smooth_loss
    )

    # 2. Context LM Loss
    context_loss = answer_loss.new_zeros(())

    if context_weight > 0:
        positions = torch.arange(
            input_ids.size(1) - 1,
            device=input_ids.device,
        ).unsqueeze(0)

        mask = positions < (lengths - 1).unsqueeze(1)

        context_hidden = hidden[:, :-1, :][mask]
        context_targets = input_ids[:, 1:][mask]

        if context_targets.numel() > 0:
            context_logits = F.linear(
                context_hidden, weight
            )

            context_loss = F.cross_entropy(
                context_logits.float(),
                context_targets,
            )

    loss = (
        answer_loss
        + context_weight * context_loss * input_ids.size(0)
    )

    choice_logits = vocab_logits.index_select(
        -1, answer_ids
    )

    return loss, choice_logits


def forward_choice_likelihood_loss(
    model,
    input_ids,
    context_lengths,
    choice_lengths,
    answers,
):
    """선택지별 평균 Log Probability 기반 Loss."""

    hidden = forward_hidden(model, input_ids)

    positions = torch.arange(
        input_ids.size(1) - 1,
        device=input_ids.device,
    )

    starts = context_lengths[:, None] - 1

    # 선택지 토큰에 해당하는 위치만 선택
    mask = (
        (positions >= starts)
        & (positions < starts + choice_lengths[:, None])
    )

    selected = model.final_norm(
        hidden[:, :-1, :][mask]
    )

    targets = input_ids[:, 1:][mask]

    logits = F.linear(
        selected,
        model.embedding.token_embedding.weight,
    )

    # 각 선택지 토큰의 Log Probability
    token_scores = -F.cross_entropy(
        logits.float(),
        targets,
        reduction="none",
    )

    rows = mask.nonzero(as_tuple=True)[0]

    # 선택지별 Log Probability 합산
    sums = token_scores.new_zeros(
        input_ids.size(0)
    ).scatter_add(0, rows, token_scores)

    # 길이 정규화 및 4개 선택지로 변환
    scores = (sums / choice_lengths).reshape(-1, 4)

    return F.cross_entropy(
        scores, answers, reduction="sum"
    )


def build_scheduler(
    optimizer,
    total_steps,
    warmup_ratio,
    learning_rate,
    min_lr,
):
    warmup_steps = int(total_steps * warmup_ratio)
    min_ratio = min_lr / learning_rate

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)

        progress = (
            (step - warmup_steps)
            / max(1, total_steps - warmup_steps)
        )

        progress = min(1.0, progress)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))

        return min_ratio + (1 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def train_finetune(
    model,
    train_loader,
    val_loader,
    answer_ids,
    config,
    device,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(config.epochs)
    accumulation = int(config.gradient_accumulation_steps)
    use_bf16 = config.mixed_precision == "bf16"

    context_weight = float(config.get("context_weight", 0.0))
    label_smoothing = float(config.get("label_smoothing", 0.0))
    choice_weight = float(config.get("choice_weight", 0.0))
    choice_microbatch = int(config.get("choice_microbatch", 2))

    model = model.to(device)
    model.gradient_checkpointing = config.gradient_checkpointing

    answer_ids = torch.as_tensor(
        answer_ids, dtype=torch.long, device=device
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=tuple(config.adam_betas),
        eps=config.adam_eps,
    )

    steps_per_epoch = math.ceil(
        len(train_loader) / accumulation
    )

    total_steps = steps_per_epoch * epochs

    scheduler = build_scheduler(
        optimizer,
        total_steps,
        config.warmup_ratio,
        config.learning_rate,
        config.min_lr,
    )

    global_step = 0
    best_accuracy = -1.0
    best_loss = float("inf")
    best_metrics = None

    log_path = output_dir / "training_log.jsonl"
    best_path = output_dir / "best_model.pt"

    start_time = time.perf_counter()

    def write_log(record):
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    # Validation 및 Best Model 저장
    def validate(epoch):
        nonlocal best_accuracy, best_loss, best_metrics

        metrics = evaluate_finetune(
            model=model,
            dataloader=val_loader,
            answer_ids=answer_ids,
            device=device,
            use_bf16=use_bf16,
        )

        record = {
            "type": "validation",
            "epoch": epoch,
            "step": global_step,
            **metrics,
        }

        write_log(record)

        accuracy = metrics["accuracy"]
        loss = metrics["loss"]

        print(
            f"[Validation] epoch={epoch} "
            f"step={global_step} "
            f"loss={loss:.6f} "
            f"accuracy={accuracy * 100:.4f}%",
            flush=True,
        )

        improved = (
            accuracy > best_accuracy
            or (
                accuracy == best_accuracy
                and loss < best_loss
            )
        )

        if improved:
            best_accuracy = accuracy
            best_loss = loss
            best_metrics = record

            save_checkpoint(
                best_path,
                {
                    "model_config": asdict(model.config),
                    "model": model.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "answer_ids": answer_ids.cpu().tolist(),
                    "validation_metrics": metrics,
                },
            )

            print(
                f"[Best Model] "
                f"accuracy={best_accuracy * 100:.4f}% "
                f"saved={best_path}",
                flush=True,
            )

    print("===== FINE-TUNING START =====")
    print("Epochs:", epochs)
    print("Total steps:", total_steps)
    print("Effective batch size:", config.batch_size * accumulation)
    print("Learning rate:", config.learning_rate)
    print("Context LM weight:", context_weight)
    print("Label smoothing:", label_smoothing)
    print("Choice Likelihood weight:", choice_weight)
    print("Choice microbatch:", choice_microbatch)
    print("Evaluation: A/B/C/D Accuracy")

    validate(epoch=0)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        window_loss = 0.0
        window_correct = 0
        window_samples = 0

        window_choice_loss = 0.0
        window_choice_count = 0

        accumulated_samples = 0

        for batch_idx, batch in enumerate(train_loader, start=1):

            input_ids = batch["input_ids"].to(device)
            label_ids = batch["label_ids"].to(device)
            lengths = batch["lengths"].to(device)
            answers = batch["answers"].to(device)

            indices = torch.arange(
                input_ids.size(0), device=device
            )

            target_ids = label_ids[indices, lengths - 1]

            # 1. Answer Loss + Context LM Loss
            with torch.autocast(
                device_type=torch.device(device).type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                loss, choice_logits = forward_context_answer_loss(
                    model,
                    input_ids,
                    lengths,
                    target_ids,
                    answer_ids,
                    context_weight,
                    label_smoothing,
                )

            loss.backward()

            batch_samples = answers.size(0)
            accumulated_samples += batch_samples
            batch_loss = loss.detach().item()

            # 2. Choice Likelihood Loss
            if choice_weight > 0:
                if "choice_answers" not in batch:
                    raise RuntimeError(
                        "Choice likelihood dataset is not enabled."
                    )

                n_valid = batch["choice_answers"].numel()

                for start in range(0, n_valid, choice_microbatch):
                    end = min(start + choice_microbatch, n_valid)

                    # 문제 하나당 선택지 4개
                    rows = slice(4 * start, 4 * end)

                    choice_inputs = batch[
                        "choice_input_ids"
                    ][rows].to(device)

                    context_lengths = batch[
                        "choice_context_lengths"
                    ][rows].to(device)

                    choice_lengths = batch[
                        "choice_lengths"
                    ][rows].to(device)

                    choice_answers = batch[
                        "choice_answers"
                    ][start:end].to(device)

                    # 불필요한 Padding 제거
                    max_len = int(
                        (context_lengths + choice_lengths).max()
                    )

                    choice_inputs = choice_inputs[
                        :, :max_len
                    ].contiguous()

                    with torch.autocast(
                        device_type=torch.device(device).type,
                        dtype=torch.bfloat16,
                        enabled=use_bf16,
                    ):
                        aux_loss = forward_choice_likelihood_loss(
                            model,
                            choice_inputs,
                            context_lengths,
                            choice_lengths,
                            choice_answers,
                        )

                    (choice_weight * aux_loss).backward()

                    aux_value = aux_loss.detach().item()

                    batch_loss += choice_weight * aux_value
                    window_choice_loss += aux_value
                    window_choice_count += end - start

                    del aux_loss

            # 3. 학습 정확도 계산
            batch_correct = (
                choice_logits.detach().argmax(dim=-1)
                == answers
            ).sum().item()

            epoch_loss += batch_loss
            epoch_correct += batch_correct
            epoch_samples += batch_samples

            window_loss += batch_loss
            window_correct += batch_correct
            window_samples += batch_samples

            # 4. Gradient Accumulation
            should_update = (
                batch_idx % accumulation == 0
                or batch_idx == len(train_loader)
            )

            if not should_update:
                continue

            for param in model.parameters():
                if param.grad is not None:
                    param.grad.div_(accumulated_samples)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.max_grad_norm,
                error_if_nonfinite=True,
            )

            current_lr = optimizer.param_groups[0]["lr"]

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            accumulated_samples = 0
            global_step += 1

            # 5. 학습 로그
            if global_step % config.log_every == 0:
                record = {
                    "type": "training",
                    "epoch": epoch,
                    "step": global_step,
                    "loss": window_loss / window_samples,
                    "accuracy": window_correct / window_samples,
                    "choice_loss": (
                        window_choice_loss / max(1, window_choice_count)
                    ),
                    "choice_coverage": (
                        window_choice_count / window_samples
                    ),
                    "lr": current_lr,
                    "grad_norm": float(grad_norm),
                    "elapsed": time.perf_counter() - start_time,
                }

                write_log(record)

                print(
                    f"[Train] epoch={epoch} "
                    f"step={global_step}/{total_steps} "
                    f"loss={record['loss']:.6f} "
                    f"acc={record['accuracy'] * 100:.2f}% "
                    f"choice_loss={record['choice_loss']:.6f} "
                    f"choice_coverage={record['choice_coverage'] * 100:.1f}% "
                    f"lr={current_lr:.3e}",
                    flush=True,
                )

                window_loss = 0.0
                window_correct = 0
                window_samples = 0
                window_choice_loss = 0.0
                window_choice_count = 0

            if global_step % config.eval_every == 0:
                validate(epoch)

        print(
            f"[Epoch {epoch}] "
            f"loss={epoch_loss / epoch_samples:.6f} "
            f"accuracy={epoch_correct / epoch_samples * 100:.4f}%",
            flush=True,
        )

        if global_step % config.eval_every != 0:
            validate(epoch)

    # 최종 결과 저장
    summary = {
        "best_accuracy": best_accuracy,
        "best_loss": best_loss,
        "best_metrics": best_metrics,
        "total_steps": global_step,
        "elapsed_seconds": time.perf_counter() - start_time,
        "checkpoint": str(best_path),
    }

    with (output_dir / "metrics.json").open("w") as f:
        json.dump(summary, f, indent=4)

    print("===== FINE-TUNING COMPLETE =====")
    print(json.dumps(summary, indent=4))

    return summary