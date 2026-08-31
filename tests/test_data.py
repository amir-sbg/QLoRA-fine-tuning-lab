import pytest
import torch

from qlora_lab.data import (
    SupervisedDataCollator,
    build_prompt,
    extract_instruction_fields,
    has_training_signal,
    normalize_text,
    split_and_limit_dataset,
    supervision_density_bucket,
    supervision_density_counts,
    tokenize_example,
    tokenized_dataset_profile,
    validate_tokenized_profile,
)


class ToyTokenizer:
    eos_token = "<eos>"
    pad_token_id = 0
    padding_side = "right"

    def __call__(
        self,
        text,
        max_length,
        truncation,
        add_special_tokens=False,
        return_tensors=None,
    ):
        pieces = text.split()
        if truncation:
            pieces = pieces[:max_length]
        ids = list(range(1, len(pieces) + 1))
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def pad(self, features, padding, return_tensors):
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_length)
            attention_mask.append(feature["attention_mask"] + [0] * pad_length)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class TinyDataset:
    column_names = ["instruction", "response"]

    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def select(self, indices):
        return TinyDataset([self.rows[index] for index in indices])

    def train_test_split(self, test_size, seed):
        cutoff = int(round(len(self.rows) * (1 - test_size)))
        return {
            "train": TinyDataset(self.rows[:cutoff]),
            "test": TinyDataset(self.rows[cutoff:]),
        }


def test_prompt_includes_context_when_available() -> None:
    prompt = build_prompt("Summarize", "A short article")

    assert "### Instruction:" in prompt
    assert "### Context:" in prompt
    assert prompt.endswith("### Response:\n")


def test_prompt_normalizes_extra_whitespace() -> None:
    prompt = build_prompt("  Summarize   this\narticle ", "  A   short\tarticle ")

    assert "Summarize this article" in prompt
    assert "A short article" in prompt


def test_extract_instruction_fields_supports_common_columns() -> None:
    instruction, context, response = extract_instruction_fields(
        {"question": "What is LoRA?", "context": "adapters", "answer": "low rank"}
    )

    assert instruction == "What is LoRA?"
    assert context == "adapters"
    assert response == "low rank"


def test_extract_instruction_fields_normalizes_text() -> None:
    instruction, context, response = extract_instruction_fields(
        {
            "instruction": "  Explain\nQLoRA ",
            "context": " adapter   training ",
            "response": " low-rank\tupdates ",
        }
    )

    assert instruction == "Explain QLoRA"
    assert context == "adapter training"
    assert response == "low-rank updates"


def test_normalize_text_handles_none() -> None:
    assert normalize_text(None) == ""


def test_training_signal_requires_instruction_and_response() -> None:
    assert has_training_signal({"instruction": "Explain", "response": "Text"})
    assert not has_training_signal({"instruction": "Explain", "response": ""})
    assert not has_training_signal({"response": "Text"})


def test_split_limits_are_applied_after_eval_split() -> None:
    dataset = TinyDataset(range(20))

    split = split_and_limit_dataset(
        dataset,
        eval_size=0.25,
        seed=7,
        max_train_samples=5,
        max_eval_samples=4,
    )

    assert len(split["train"]) == 5
    assert len(split["eval"]) == 4


def test_tokenize_masks_prompt_tokens() -> None:
    encoded = tokenize_example(
        {"instruction": "Say hello", "response": "hello there"},
        ToyTokenizer(),
        max_seq_length=64,
    )

    assert len(encoded["input_ids"]) == len(encoded["labels"])
    assert -100 in encoded["labels"]
    assert encoded["labels"][-1] != -100


def test_supervised_collator_preserves_label_mask() -> None:
    tokenizer = ToyTokenizer()
    collator = SupervisedDataCollator(tokenizer)
    batch = collator(
        [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
            {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]},
        ]
    )

    assert batch["input_ids"].shape == (2, 3)
    assert batch["labels"].tolist() == [[-100, 2, 3], [-100, 5, -100]]


def test_dataset_profile_reports_supervised_ratio() -> None:
    profile = tokenized_dataset_profile(
        [
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
            {"input_ids": [4, 5], "labels": [-100, 5]},
        ]
    )

    assert profile["examples"] == 2
    assert profile["max_input_tokens"] == 3
    assert profile["supervised_token_ratio"] == 0.6


def test_supervision_density_bucket_tracks_loss_signal() -> None:
    assert supervision_density_bucket(input_tokens=32, supervised_tokens=0) == "empty"
    assert supervision_density_bucket(input_tokens=100, supervised_tokens=2) == "under_5_pct"
    assert supervision_density_bucket(input_tokens=100, supervised_tokens=12) == "5_to_20_pct"
    assert supervision_density_bucket(input_tokens=100, supervised_tokens=45) == "over_20_pct"


def test_dataset_profile_reports_density_buckets() -> None:
    profile = tokenized_dataset_profile(
        [
            {"input_ids": list(range(32)), "labels": [-100] * 32},
            {"input_ids": list(range(100)), "labels": [-100] * 98 + [1, 2]},
            {"input_ids": list(range(100)), "labels": [-100] * 88 + list(range(12))},
            {"input_ids": list(range(100)), "labels": [-100] * 55 + list(range(45))},
        ]
    )

    assert profile["min_supervised_tokens"] == 0
    assert profile["max_supervised_tokens"] == 45
    assert profile["supervision_density_buckets"] == supervision_density_counts(
        [32, 100, 100, 100],
        [0, 2, 12, 45],
    )


def test_profile_validation_rejects_empty_split() -> None:
    profile = tokenized_dataset_profile([])

    with pytest.raises(ValueError, match="no tokenized examples"):
        validate_tokenized_profile(profile, "train")


def test_profile_validation_rejects_low_supervised_ratio() -> None:
    profile = {
        "examples": 1,
        "avg_input_tokens": 100.0,
        "max_input_tokens": 100,
        "avg_supervised_tokens": 1.0,
        "supervised_token_ratio": 0.005,
    }

    with pytest.raises(ValueError, match="very few supervised tokens"):
        validate_tokenized_profile(profile, "train")
