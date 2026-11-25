import random
import re
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from datasets import load_dataset


ANSWER_LABELS = ("A", "B", "C", "D")


def load_finetune_data(
    path="/data/finetuning-dataset/train.parquet",
):
    return load_dataset(
        "parquet",
        data_files=path,
        split="train",
    )


def group_split(dataset, val_ratio=0.05, seed=42):
    """동일 질문이 Train과 Validation에 동시에 포함되지 않도록 분리."""

    groups = defaultdict(list)

    for idx, row in enumerate(dataset):
        question = " ".join(row["question"].split())
        groups[question].append(idx)

    keys = list(groups.keys())

    rng = random.Random(seed)
    rng.shuffle(keys)

    val_target = round(len(dataset) * val_ratio)

    train_indices = []
    val_indices = []

    for key in keys:

        if len(val_indices) < val_target:
            val_indices.extend(groups[key])
        else:
            train_indices.extend(groups[key])

    return (
        dataset.select(train_indices),
        dataset.select(val_indices),
    )


def build_context(question, choices):
    """교수님이 지정한 입력 형식."""

    question = question.strip()

    choices_str = " ".join(
        f"({ANSWER_LABELS[i]}) {choice}"
        for i, choice in enumerate(choices)
    )

    context = f"{question}\n{choices_str}\nAnswer: "

    return context


def get_answer_ids(tokenizer):
    """A, B, C, D의 단일 토큰 ID 반환."""

    answer_ids = []

    for label in ANSWER_LABELS:

        ids = tokenizer.encode(label)

        if len(ids) != 1:
            raise ValueError(
                f"Answer {label} is not a single token."
            )

        answer_ids.append(ids[0])

    return answer_ids


def can_shuffle(question, choices):
    """선택지 순서에 의존하는 문제는 셔플하지 않는다."""

    text = " ".join([question, *choices]).lower()

    patterns = [
        r"\b(?:all|none|both|any) of the above\b",
        r"\b(?:all|none|both|any) of these\b",
        r"\b(?:all|none|both) of the following\b",
        r"\b(?:first|second|third|fourth) option\b",
        r"\b(?:option|choice|answer)s?\s+[abcd]\b",
        r"\b[abcd]\s*(?:and|or|&)\s*[abcd]\b",
        r"\b(?:above|below) options?\b",
        r"\([abcd]\)",
    ]

    return not any(
        re.search(pattern, text)
        for pattern in patterns
    )


def shuffle_options(choices, answer):

    indices = list(range(4))
    random.shuffle(indices)

    shuffled_choices = [
        choices[i] for i in indices
    ]

    shuffled_answer = indices.index(answer)

    return shuffled_choices, shuffled_answer


# ==========================================
# Choice Likelihood 입력 생성
# ==========================================

def encode_choice_candidates(
    question,
    choices,
    tokenizer,
    max_seq_len=1024,
):
    """
    질문과 개별 선택지로 구성된 입력 4개 생성.

    예:
        Question: What is the capital of France?
        Answer: Paris

    정답 위치와 관계없이 모든 선택지를 동일하게 처리한다.
    """

    question = question.strip()

    context = f"Question: {question}\nAnswer:"

    context_ids = [
        tokenizer.bos_id
    ] + tokenizer.encode(context)

    encoded_choices = [
        tokenizer.encode(
            " " + str(choice).strip()
        )
        for choice in choices
    ]

    # 빈 선택지가 존재하면 보조 학습에서 제외
    if any(
        len(ids) == 0
        for ids in encoded_choices
    ):
        return None

    max_choice_len = max(
        len(ids) for ids in encoded_choices
    )

    # 모든 선택지에 동일한 질문 길이를 사용하기 위한 검사
    if len(context_ids) + max_choice_len > max_seq_len:
        return None

    choice_input_ids = []
    choice_context_lengths = []
    choice_lengths = []

    for choice_ids in encoded_choices:

        full_ids = context_ids + choice_ids

        choice_input_ids.append(
            torch.tensor(
                full_ids,
                dtype=torch.long,
            )
        )

        choice_context_lengths.append(
            len(context_ids)
        )

        choice_lengths.append(
            len(choice_ids)
        )

    return {
        "choice_input_ids": choice_input_ids,
        "choice_context_lengths": choice_context_lengths,
        "choice_lengths": choice_lengths,
    }


# ==========================================
# Finetune Dataset
# ==========================================

class FinetuneDataset(Dataset):

    def __init__(
        self,
        dataset,
        tokenizer,
        max_seq_len=1024,
        shuffle_choices=False,
        enable_choice_likelihood=False,
    ):

        self.samples = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.shuffle_choices = shuffle_choices

        # Choice Likelihood 보조 학습 활성화
        self.enable_choice_likelihood = (
            enable_choice_likelihood
        )

        self.truncated_count = 0
        self.shuffleable_count = 0

        # 보조 학습 통계
        self.choice_valid_count = 0
        self.choice_skipped_count = 0

        for row in dataset:

            question = row["question"]
            choices = row["choices"]
            answer = int(row["answer"])

            if len(choices) != 4 or not 0 <= answer < 4:
                raise ValueError("Invalid QA sample.")

            order_independent = can_shuffle(
                question,
                choices,
            )

            shuffleable = (
                self.shuffle_choices
                and order_independent
            )

            if shuffleable:
                self.shuffleable_count += 1

            # 기존 교수님 형식 토큰화
            sample, truncated = self.encode_sample(
                question,
                choices,
                answer,
            )

            if truncated:
                self.truncated_count += 1

            # 선택지 보조 학습 데이터 생성
            choice_data = None

            if self.enable_choice_likelihood:

                # 선택지 순서나 다른 선택지에 의존하는 문제 제외
                if order_independent:

                    choice_data = encode_choice_candidates(
                        question=question,
                        choices=choices,
                        tokenizer=self.tokenizer,
                        max_seq_len=self.max_seq_len,
                    )

                if choice_data is not None:

                    self.choice_valid_count += 1

                    sample.update(choice_data)
                    sample["choice_valid"] = True

                else:

                    self.choice_skipped_count += 1
                    sample["choice_valid"] = False

            self.samples.append({
                "question": question,
                "choices": choices,
                "answer": answer,
                "shuffleable": shuffleable,
                "encoded": sample,
                "choice_data": choice_data,
            })

    def encode_sample(
        self,
        question,
        choices,
        answer,
    ):

        tokenizer = self.tokenizer

        context = build_context(
            question,
            choices,
        )

        context_ids = tokenizer.encode(context)

        max_context_len = self.max_seq_len - 1
        truncated = False

        if len(context_ids) > max_context_len:

            truncated = True

            empty_context = build_context(
                "",
                choices,
            )

            fixed_len = len(
                tokenizer.encode(empty_context)
            )

            if fixed_len > max_context_len:

                raise ValueError(
                    "Choices exceed maximum context length."
                )

            question_ids = tokenizer.encode(question)

            budget = max_context_len - fixed_len

            while True:

                shortened = tokenizer.decode(
                    question_ids[:budget],
                    errors="ignore",
                )

                context = build_context(
                    shortened,
                    choices,
                )

                context_ids = tokenizer.encode(context)

                if len(context_ids) <= max_context_len:
                    break

                if budget == 0:

                    raise ValueError(
                        "Unable to truncate context."
                    )

                budget = max(
                    0,
                    budget - max(
                        1,
                        len(context_ids) - max_context_len,
                    ),
                )

        input_ids = [
            tokenizer.bos_id
        ] + context_ids

        answer_id = tokenizer.encode(
            ANSWER_LABELS[answer]
        )

        label_ids = (
            [tokenizer.pad_id] * len(context_ids)
            + answer_id
        )

        assert len(input_ids) == len(label_ids)

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "label_ids": torch.tensor(
                label_ids,
                dtype=torch.long,
            ),
            "length": len(input_ids),
            "answer": answer,
        }, truncated

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        # Validation 또는 셔플 불가능한 문제
        if not sample["shuffleable"]:

            return sample["encoded"]

        # 접근할 때마다 새로운 선택지 순서 생성
        indices = list(range(4))
        random.shuffle(indices)

        choices = [
            sample["choices"][i]
            for i in indices
        ]

        answer = indices.index(
            sample["answer"]
        )

        encoded, _ = self.encode_sample(
            sample["question"],
            choices,
            answer,
        )

        # Choice Likelihood 입력에도 동일한 순서 적용
        if self.enable_choice_likelihood:

            choice_data = sample["choice_data"]

            if choice_data is not None:

                encoded["choice_valid"] = True

                encoded["choice_input_ids"] = [
                    choice_data["choice_input_ids"][i]
                    for i in indices
                ]

                encoded["choice_context_lengths"] = [
                    choice_data["choice_context_lengths"][i]
                    for i in indices
                ]

                encoded["choice_lengths"] = [
                    choice_data["choice_lengths"][i]
                    for i in indices
                ]

            else:

                encoded["choice_valid"] = False

        return encoded


# ==========================================
# Finetune Collator
# ==========================================

class FinetuneCollator:
    """기존 Right Padding 및 Choice Likelihood 배치 생성."""

    def __init__(self, pad_id):

        self.pad_id = pad_id

    def __call__(self, batch):

        lengths = torch.tensor(
            [sample["length"] for sample in batch],
            dtype=torch.long,
        )

        max_len = int(lengths.max())

        input_ids = torch.full(
            (len(batch), max_len),
            self.pad_id,
            dtype=torch.long,
        )

        label_ids = torch.full(
            (len(batch), max_len),
            self.pad_id,
            dtype=torch.long,
        )

        for i, sample in enumerate(batch):

            ids = sample["input_ids"]
            labels = sample["label_ids"]

            input_ids[i, :len(ids)] = ids
            label_ids[i, :len(labels)] = labels

        answers = torch.tensor(
            [sample["answer"] for sample in batch],
            dtype=torch.long,
        )

        result = {
            "input_ids": input_ids,
            "label_ids": label_ids,
            "lengths": lengths,
            "answers": answers,
        }

        # Validation은 기존 데이터만 반환
        if not any(
            "choice_valid" in sample
            for sample in batch
        ):
            return result

        # ==================================
        # Choice Likelihood Batch
        # ==================================

        valid_indices = [
            i
            for i, sample in enumerate(batch)
            if sample.get("choice_valid", False)
        ]

        result["choice_valid_indices"] = torch.tensor(
            valid_indices,
            dtype=torch.long,
        )

        num_valid = len(valid_indices)

        if num_valid == 0:

            result["choice_input_ids"] = torch.empty(
                (0, 0),
                dtype=torch.long,
            )

            result["choice_context_lengths"] = torch.empty(
                (0,),
                dtype=torch.long,
            )

            result["choice_lengths"] = torch.empty(
                (0,),
                dtype=torch.long,
            )

            result["choice_answers"] = torch.empty(
                (0,),
                dtype=torch.long,
            )

            return result

        candidates = []
        context_lengths = []
        choice_lengths = []
        choice_answers = []

        for idx in valid_indices:

            sample = batch[idx]

            candidates.extend(
                sample["choice_input_ids"]
            )

            context_lengths.extend(
                sample["choice_context_lengths"]
            )

            choice_lengths.extend(
                sample["choice_lengths"]
            )

            choice_answers.append(
                sample["answer"]
            )

        max_choice_seq_len = max(
            len(ids)
            for ids in candidates
        )

        choice_input_ids = torch.full(
            (
                num_valid * 4,
                max_choice_seq_len,
            ),
            self.pad_id,
            dtype=torch.long,
        )

        for i, ids in enumerate(candidates):

            choice_input_ids[i, :len(ids)] = ids

        result["choice_input_ids"] = choice_input_ids

        result["choice_context_lengths"] = torch.tensor(
            context_lengths,
            dtype=torch.long,
        )

        result["choice_lengths"] = torch.tensor(
            choice_lengths,
            dtype=torch.long,
        )

        result["choice_answers"] = torch.tensor(
            choice_answers,
            dtype=torch.long,
        )

        return result