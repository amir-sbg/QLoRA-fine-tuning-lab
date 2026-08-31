import pytest

from qlora_lab.config import QLoRAConfig
from qlora_lab.preflight import (
    build_preflight_report,
    effective_batch_size,
    estimate_token_budget,
    estimate_update_steps,
)


def test_effective_batch_size_uses_accumulation_and_world_size() -> None:
    config = QLoRAConfig(batch_size=2, gradient_accumulation_steps=8)

    assert effective_batch_size(config, world_size=4) == 64


def test_step_estimate_rounds_up_partial_epochs() -> None:
    config = QLoRAConfig(batch_size=2, gradient_accumulation_steps=4, epochs=1.5)
    steps = estimate_update_steps(65, config)

    assert steps["effective_batch_size"] == 8
    assert steps["steps_per_epoch"] == 9
    assert steps["estimated_total_steps"] == 14
    assert steps["estimated_warmup_steps"] == 1
    assert steps["estimated_decay_steps"] == 13


def test_token_budget_uses_effective_batch_and_context_length() -> None:
    config = QLoRAConfig(
        batch_size=2,
        gradient_accumulation_steps=4,
        max_seq_length=128,
        epochs=1.5,
    )
    budget = estimate_token_budget(65, config, world_size=2)

    assert budget["tokens_per_device_batch"] == 256
    assert budget["tokens_per_update"] == 2048
    assert budget["max_seen_tokens"] == 98 * 128
    assert budget["warmup_seen_tokens"] == 2048


def test_preflight_report_includes_memory_estimate() -> None:
    report = build_preflight_report(
        QLoRAConfig(lora_r=8, lora_alpha=16),
        train_examples=100,
        base_parameters=10_000,
    )

    assert report["qlora"]["lora_scale"] == 2.0
    assert report["optimizer"]["learning_rate"] == QLoRAConfig.learning_rate
    assert report["steps"]["train_examples"] == 100
    assert report["token_budget"]["max_seq_length"] == QLoRAConfig.max_seq_length
    assert report["memory_estimate"]["base_parameters"] == 10_000


def test_preflight_rejects_bad_world_size() -> None:
    with pytest.raises(ValueError, match="world_size"):
        effective_batch_size(QLoRAConfig(), world_size=0)
