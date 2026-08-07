from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .config import QLoRAConfig


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def build_prompt(instruction: str, context: str | None = None) -> str:
    instruction = normalize_text(instruction)
    context = normalize_text(context)
    parts = ["### Instruction:", instruction]
    if context:
        parts.extend(["", "### Context:", context])
    parts.extend(["", "### Response:"])
    return "\n".join(parts) + "\n"


def extract_instruction_fields(example: dict[str, Any]) -> tuple[str, str, str]:
    instruction = (
        example.get("instruction")
        or example.get("prompt")
        or example.get("question")
        or example.get("input")
        or ""
    )
    context = example.get("context") or example.get("input") or ""
    response = (
        example.get("response")
        or example.get("output")
        or example.get("answer")
        or example.get("completion")
        or ""
    )
    return normalize_text(instruction), normalize_text(context), normalize_text(response)


def has_training_signal(example: dict[str, Any]) -> bool:
    instruction, _, response = extract_instruction_fields(example)
    return bool(instruction.strip() and response.strip())


def format_training_text(example: dict[str, Any], eos_token: str = "") -> tuple[str, str]:
    instruction, context, response = extract_instruction_fields(example)
    prompt = build_prompt(instruction, context)
    return prompt, prompt + response.strip() + eos_token


def tokenize_example(
    example: dict[str, Any],
    tokenizer: Any,
    max_seq_length: int,
) -> dict[str, list[int]]:
    prompt, text = format_training_text(example, eos_token=tokenizer.eos_token or "")
    encoded = tokenizer(
        text,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False,
    )
    prompt_ids = tokenizer(
        prompt,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False,
    )["input_ids"]
    labels = list(encoded["input_ids"])
    prompt_length = min(len(prompt_ids), len(labels))
    labels[:prompt_length] = [-100] * prompt_length
    if labels and all(value == -100 for value in labels):
        labels[-1] = encoded["input_ids"][-1]
    encoded["labels"] = labels
    return encoded


@dataclass
class SupervisedDataCollator:
    tokenizer: Any
    label_pad_token_id: int = -100

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        labels = [list(feature["labels"]) for feature in features]
        model_inputs = [
            {key: value for key, value in feature.items() if key != "labels"}
            for feature in features
        ]
        batch = self.tokenizer.pad(
            model_inputs,
            padding=True,
            return_tensors="pt",
        )
        max_length = int(batch["input_ids"].shape[1])
        padded_labels = []
        for label in labels:
            pad_length = max_length - len(label)
            if pad_length < 0:
                raise ValueError("label length exceeds padded input length")
            padding = [self.label_pad_token_id] * pad_length
            if getattr(self.tokenizer, "padding_side", "right") == "left":
                padded_labels.append(padding + label)
            else:
                padded_labels.append(label + padding)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def tokenized_dataset_profile(dataset: Any) -> dict[str, float | int]:
    rows = list(dataset)
    if not rows:
        return {
            "examples": 0,
            "avg_input_tokens": 0.0,
            "max_input_tokens": 0,
            "avg_supervised_tokens": 0.0,
            "supervised_token_ratio": 0.0,
        }

    input_lengths = [len(row["input_ids"]) for row in rows]
    supervised_lengths = [
        sum(1 for token in row["labels"] if token != -100)
        for row in rows
    ]
    total_tokens = sum(input_lengths)
    return {
        "examples": len(rows),
        "avg_input_tokens": round(sum(input_lengths) / len(rows), 3),
        "max_input_tokens": max(input_lengths),
        "avg_supervised_tokens": round(sum(supervised_lengths) / len(rows), 3),
        "supervised_token_ratio": round(
            sum(supervised_lengths) / max(total_tokens, 1),
            4,
        ),
    }


def validate_tokenized_profile(
    profile: dict[str, float | int],
    split_name: str,
    min_supervised_ratio: float = 0.01,
) -> None:
    if profile["examples"] < 1:
        raise ValueError(f"{split_name} split has no tokenized examples")
    if profile["avg_supervised_tokens"] <= 0:
        raise ValueError(f"{split_name} split has no supervised response tokens")
    if profile["supervised_token_ratio"] < min_supervised_ratio:
        raise ValueError(
            f"{split_name} split has very few supervised tokens; "
            "check prompt masking and max_seq_length"
        )


def _limit_split(dataset: Any, max_samples: int | None) -> Any:
    if max_samples is None:
        return dataset
    return dataset.select(range(min(max_samples, len(dataset))))


def prepare_datasets(config: QLoRAConfig, tokenizer: Any) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets to load fine-tuning data.") from exc

    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split=config.dataset_split,
    )
    dataset = dataset.filter(
        has_training_signal,
        desc="Filtering empty instruction rows",
    )
    dataset = dataset.shuffle(seed=config.seed)
    dataset = _limit_split(dataset, config.max_train_samples)
    split = dataset.train_test_split(test_size=config.eval_size, seed=config.seed)
    train_dataset = _limit_split(split["train"], config.max_train_samples)
    eval_dataset = _limit_split(split["test"], config.max_eval_samples)

    def tokenize(row: dict[str, Any]) -> dict[str, list[int]]:
        return tokenize_example(row, tokenizer, config.max_seq_length)

    train_dataset = train_dataset.map(
        tokenize,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train examples",
    )
    eval_dataset = eval_dataset.map(
        tokenize,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing eval examples",
    )
    return {"train": train_dataset, "eval": eval_dataset}
