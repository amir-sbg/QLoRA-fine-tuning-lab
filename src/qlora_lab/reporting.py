from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("report must be a JSON object")
    return payload


def run_card(train_summary: dict[str, Any]) -> dict[str, Any]:
    config = train_summary.get("config", {})
    train_profile = train_summary.get("data_profile", {}).get("train", {})
    eval_profile = train_summary.get("data_profile", {}).get("eval", {})
    parameters = train_summary.get("trainable_parameters", {})
    return {
        "model_name": config.get("model_name"),
        "dataset_name": config.get("dataset_name"),
        "lora_r": config.get("lora_r"),
        "lora_alpha": config.get("lora_alpha"),
        "lora_scale": config.get("lora_scale"),
        "max_seq_length": config.get("max_seq_length"),
        "train_examples": train_profile.get("examples"),
        "eval_examples": eval_profile.get("examples"),
        "supervised_token_ratio": train_profile.get("supervised_token_ratio"),
        "trainable_percent": parameters.get("trainable_percent"),
    }


def compare_run_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        cards,
        key=lambda row: (
            row.get("lora_r") is None,
            row.get("lora_r") or 0,
            row.get("max_seq_length") or 0,
        ),
    )
