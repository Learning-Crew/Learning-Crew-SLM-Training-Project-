
import json
import time

import torch
from torch.utils.data import DataLoader

from configs.model_config import ModelConfig
from configs.tokenizer_config import (
    TokenizerConfig,
    PretrainDatasetConfig,
)
from tokenizer.bpe_tokenizer import BPETokenizer
from model.gpt import GPTModel

from data.finetune_dataset import (
    load_finetune_data,
    group_split,
    FinetuneDataset,
    FinetuneCollator,
    get_answer_ids,
)

from evaluation.finetune_eval import (
    evaluate_finetune,
)


CHECKPOINT_PATH = (
    "/home/team3/models/pretrained_300m_1.5ep.pt"
)

DEVICE = "cuda"
BATCH_SIZE = 8


def load_pretrained_model(path):
    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    config = ModelConfig(**state["model_config"])

    model = GPTModel(config)

    model.load_state_dict(
        state["model"],
        strict=True,
    )

    print("Pretrained steps:",
          state["total_completed_steps"])

    del state

    return model.to(DEVICE)


@torch.inference_mode()
def test_padding(model, dataset, tokenizer, answer_ids):
    model.eval()

    ids = dataset[0]["input_ids"].unsqueeze(0)
    length = ids.size(1)

    pad_length = min(
        32, model.max_seq_len - length
    )

    if pad_length == 0:
        raise ValueError("No space for padding test.")

    padding = torch.full(
        (1, pad_length),
        tokenizer.pad_id,
        dtype=torch.long,
    )

    padded_ids = torch.cat(
        [ids, padding], dim=1
    )

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        original = model(
            ids.to(DEVICE)
        )[0, length - 1, answer_ids].float()

        padded = model(
            padded_ids.to(DEVICE)
        )[0, length - 1, answer_ids].float()

    print("\n===== PADDING TEST =====")
    print("Original:", original.tolist())
    print("Padded:", padded.tolist())

    print(
        "Max difference:",
        (original - padded).abs().max().item(),
    )

    print(
        "Same prediction:",
        original.argmax().item()
        == padded.argmax().item(),
    )

    print(
        "Close logits:",
        torch.allclose(
            original,
            padded,
            atol=0.1,
            rtol=0.01,
        ),
    )


def main():
    start = time.perf_counter()

    print("===== LOAD MODEL =====")
    model = load_pretrained_model(CHECKPOINT_PATH)

    print(
        "Parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    print("\n===== LOAD TOKENIZER =====")

    tokenizer = BPETokenizer(
        TokenizerConfig(
            vocab_path="/home/team3/tokenizer_32k/vocab.json",
            merges_path="/home/team3/tokenizer_32k/merges.json",
        ),
        PretrainDatasetConfig(),
    )

    answer_ids = get_answer_ids(tokenizer)

    print("Answer IDs:", answer_ids)

    print("\n===== LOAD VALIDATION DATA =====")

    ds = load_finetune_data()

    _, val_ds = group_split(
        ds,
        val_ratio=0.05,
        seed=42,
    )

    val_data = FinetuneDataset(
        val_ds,
        tokenizer,
        max_seq_len=model.max_seq_len,
    )

    print("Validation samples:", len(val_data))

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=FinetuneCollator(
            tokenizer.pad_id
        ),
    )

    test_padding(
        model,
        val_data,
        tokenizer,
        answer_ids,
    )

    print("\n===== BASELINE EVALUATION =====")

    metrics = evaluate_finetune(
        model=model,
        dataloader=val_loader,
        answer_ids=answer_ids,
        device=DEVICE,
        use_bf16=True,
    )

    print(json.dumps(metrics, indent=4))

    print(
        f"\nBaseline Accuracy: "
        f"{metrics['accuracy'] * 100:.4f}%"
    )

    print(
        "Elapsed:",
        round(time.perf_counter() - start, 2),
        "s",
    )

    with open(
        "finetune_baseline_metrics.json",
        "w",
    ) as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
