import pytest

from qlora_lab.config import QLoRAConfig
from qlora_lab.preflight import (
    build_preflight_report,
    effective_batch_size,
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


def test_preflight_report_includes_memory_estimate() -> None:
    report = build_preflight_report(
        QLoRAConfig(lora_r=8, lora_alpha=16),
        train_examples=100,
        base_parameters=10_000,
    )

    assert report["qlora"]["lora_scale"] == 2.0
    assert report["steps"]["train_examples"] == 100
    assert report["memory_estimate"]["base_parameters"] == 10_000


def test_preflight_rejects_bad_world_size() -> None:
    with pytest.raises(ValueError, match="world_size"):
        effective_batch_size(QLoRAConfig(), world_size=0)
