import json

import pytest

from qlora_lab.reporting import compare_run_cards, load_json_report, run_card


def test_run_card_keeps_core_training_fields() -> None:
    card = run_card(
        {
            "config": {
                "model_name": "tiny",
                "dataset_name": "instructions",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_scale": 2.0,
                "max_seq_length": 256,
            },
            "data_profile": {
                "train": {"examples": 100, "supervised_token_ratio": 0.42},
                "eval": {"examples": 20},
            },
            "trainable_parameters": {"trainable_percent": 1.2},
        }
    )

    assert card["lora_r"] == 8
    assert card["train_examples"] == 100
    assert card["supervised_token_ratio"] == 0.42


def test_compare_run_cards_orders_by_rank_and_context() -> None:
    rows = compare_run_cards(
        [
            {"lora_r": 16, "max_seq_length": 256},
            {"lora_r": 8, "max_seq_length": 512},
            {"lora_r": 8, "max_seq_length": 256},
        ]
    )

    assert rows[0] == {"lora_r": 8, "max_seq_length": 256}
    assert rows[-1] == {"lora_r": 16, "max_seq_length": 256}


def test_load_json_report_requires_object(tmp_path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_json_report(path)
