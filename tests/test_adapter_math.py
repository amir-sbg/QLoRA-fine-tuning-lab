import pytest
import torch

from qlora_lab.adapter_math import lora_update_diagnostics


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
