from __future__ import annotations

from typing import Any

from .config import QLoRAConfig


def build_prompt(instruction: str, context: str | None = None) -> str:
    instruction = instruction.strip()
    context = (context or "").strip()
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
    return str(instruction), str(context), str(response)


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
