from pathlib import Path

import pytest

from qlora_lab.config import QLoRAConfig


def test_config_reports_lora_scale() -> None:
    config = QLoRAConfig(
        lora_r=8,
        lora_alpha=24,
        output_dir=Path("out"),
        resume_from_checkpoint=Path("ckpt"),
    )
    payload = config.to_dict()

    assert config.lora_scale == 3.0
    assert payload["lora_scale"] == 3.0
    assert payload["output_dir"] == "out"
    assert payload["resume_from_checkpoint"] == "ckpt"


def test_config_rejects_bad_rank() -> None:
    with pytest.raises(ValueError, match="lora_r"):
        QLoRAConfig(lora_r=0)


def test_config_rejects_invalid_gradient_clip() -> None:
    with pytest.raises(ValueError, match="max_grad_norm"):
        QLoRAConfig(max_grad_norm=0)
