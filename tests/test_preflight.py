import pytest

from qlora_lab.config import QLoRAConfig
from qlora_lab.preflight import (
    build_preflight_report,
    effective_batch_size,
    estimate_token_budget,
    estimate_hidden_state_memory_mb,
    estimate_adapter_optimizer_memory_mb,
    estimate_training_memory_budget_mb,
    estimate_update_steps,
    learning_rate_preview,
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


def test_hidden_state_memory_estimate_reflects_checkpointing() -> None:
    estimate = estimate_hidden_state_memory_mb(
        batch_size=2,
        sequence_length=512,
        hidden_size=1024,
        layers=16,
    )

    assert estimate["estimated_checkpoint_count"] == 4
    assert estimate["checkpointed_hidden_state_mb"] < estimate["uncheckpointed_hidden_state_mb"]


def test_adapter_optimizer_memory_includes_gradients_and_moments() -> None:
    estimate = estimate_adapter_optimizer_memory_mb(1_048_576)

    assert estimate["adapter_parameters_mb"] == pytest.approx(2.0)
    assert estimate["adapter_gradients_mb"] == pytest.approx(2.0)
    assert estimate["optimizer_states_mb"] == pytest.approx(8.0)
    assert estimate["adapter_training_state_mb"] == pytest.approx(12.0)


def test_preflight_report_includes_memory_estimate() -> None:
    report = build_preflight_report(
        QLoRAConfig(lora_r=8, lora_alpha=16),
        train_examples=100,
        base_parameters=10_000,
        hidden_size=128,
        layers=4,
        trainable_parameters=4096,
    )

    assert report["qlora"]["lora_scale"] == 2.0
    assert report["optimizer"]["learning_rate"] == QLoRAConfig.learning_rate
    assert report["data_plan"]["max_eval_samples"] == QLoRAConfig.max_eval_samples
    assert report["steps"]["train_examples"] == 100
    assert report["token_budget"]["max_seq_length"] == QLoRAConfig.max_seq_length
    assert report["lr_preview"][0]["step"] == 1
    assert report["memory_estimate"]["base_parameters"] == 10_000
    assert report["activation_memory_estimate"]["layers"] == 4
    assert report["adapter_training_memory_estimate"]["trainable_parameters"] == 4096
    assert report["training_memory_budget"]["estimated_total_mb"] > 0
    assert any("Token exposure is low" in warning for warning in report["warnings"])


def test_preflight_warns_about_small_eval_cap() -> None:
    report = build_preflight_report(QLoRAConfig(max_eval_samples=16))

    assert any("Evaluation cap is small" in warning for warning in report["warnings"])


def test_learning_rate_preview_tracks_warmup_and_decay() -> None:
    preview = learning_rate_preview(
        total_steps=10,
        warmup_steps=2,
        learning_rate=1e-4,
    )

    by_step = {row["step"]: row["lr"] for row in preview}
    assert by_step[1] == 0.00005
    assert by_step[2] == 0.0001
    assert by_step[10] == 0.0


def test_preflight_rejects_bad_world_size() -> None:
    with pytest.raises(ValueError, match="world_size"):
        effective_batch_size(QLoRAConfig(), world_size=0)


def test_training_memory_budget_keeps_components_visible() -> None:
    budget = estimate_training_memory_budget_mb(
        base_parameters=1_000_000,
        trainable_parameters=100_000,
        checkpointed_activation_mb=32.0,
        workspace_fraction=0.2,
    )

    subtotal = (
        budget["quantized_base_mb"]
        + budget["adapter_training_state_mb"]
        + budget["checkpointed_activation_mb"]
    )
    assert budget["workspace_reserve_mb"] == pytest.approx(subtotal * 0.2, abs=0.002)
    assert budget["estimated_total_mb"] > subtotal
