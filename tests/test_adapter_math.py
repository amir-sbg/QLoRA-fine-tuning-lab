import pytest
import torch

from qlora_lab.adapter_math import (
    lora_scaling,
    lora_spectrum_report,
    lora_update_diagnostics,
    merge_lora_weight,
)


def test_lora_update_diagnostics_match_diagonal_update() -> None:
    lora_a = torch.eye(2)
    lora_b = torch.eye(2)

    report = lora_update_diagnostics(
        lora_a,
        lora_b,
        alpha=4,
        base_weight=torch.eye(2),
    )

    assert report["scale"] == 2.0
    assert report["update_spectral_norm"] == pytest.approx(2.0)
    assert report["update_stable_rank"] == pytest.approx(2.0)
    assert report["update_numerical_rank"] == 2
    assert report["relative_update_norm"] == pytest.approx(2.0)
    assert report["base_update_cosine"] == pytest.approx(1.0)
    assert report["base_update_angle_degrees"] == pytest.approx(0.0)
    assert report["base_update_sign_agreement"] == pytest.approx(1.0)


def test_lora_update_diagnostics_validate_factor_shapes() -> None:
    with pytest.raises(ValueError, match="rank dimensions"):
        lora_update_diagnostics(torch.zeros(2, 4), torch.zeros(3, 3), alpha=4)
    with pytest.raises(ValueError, match="update shape"):
        lora_update_diagnostics(
            torch.zeros(2, 4),
            torch.zeros(3, 2),
            alpha=4,
            base_weight=torch.zeros(4, 3),
        )


def test_merge_matches_unmerged_linear_projection() -> None:
    base = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    lora_a = torch.tensor([[1.0, -1.0]])
    lora_b = torch.tensor([[0.5], [1.0]])
    inputs = torch.tensor([[2.0, 3.0]])

    merged = merge_lora_weight(base, lora_a, lora_b, alpha=2.0)
    unmerged_output = inputs @ base.T + lora_scaling(2.0, 1) * (inputs @ lora_a.T) @ lora_b.T

    torch.testing.assert_close(inputs @ merged.T, unmerged_output)


def test_rank_stabilized_diagnostics_use_sqrt_rank() -> None:
    report = lora_update_diagnostics(
        torch.eye(4), torch.eye(4), alpha=8, use_rslora=True
    )

    assert report["scale"] == 4.0
    assert report["scaling_rule"] == "alpha/sqrt(r)"


def test_lora_spectrum_reports_update_energy_concentration() -> None:
    report = lora_spectrum_report(
        torch.eye(2),
        torch.diag(torch.tensor([3.0, 1.0])),
        alpha=2,
    )

    assert report["top1_energy_fraction"] == pytest.approx(0.9)
    assert report["rank_90_percent_energy"] == 1
    assert 1.0 < report["entropy_effective_rank"] < 2.0
    assert report["energy_participation_rank"] == pytest.approx(1 / 0.82)
    assert report["active_rank"] == 2
    assert report["active_rank_fraction"] == 1.0
    assert report["active_condition_number"] == pytest.approx(3.0)


def test_lora_spectrum_detects_unused_configured_rank() -> None:
    report = lora_spectrum_report(
        torch.eye(3),
        torch.diag(torch.tensor([2.0, 0.0, 0.0])),
        alpha=3,
    )

    assert report["configured_rank"] == 3
    assert report["active_rank"] == 1
    assert report["active_rank_fraction"] == pytest.approx(1 / 3)


def test_lora_diagnostics_expose_opposing_update_direction() -> None:
    report = lora_update_diagnostics(
        torch.eye(2),
        -torch.eye(2),
        alpha=2,
        base_weight=torch.eye(2),
    )

    assert report["base_update_cosine"] == pytest.approx(-1.0)
    assert report["base_update_angle_degrees"] == pytest.approx(180.0)
    assert report["base_update_sign_agreement"] == pytest.approx(0.0)
