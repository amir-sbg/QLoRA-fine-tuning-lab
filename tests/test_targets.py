from __future__ import annotations

import pytest
import torch

from qlora_lab.targets import (
    collect_linear_module_names,
    target_module_matches,
    validate_target_modules,
)


class TinyDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4)
        self.mlp = torch.nn.ModuleDict(
            {
                "up_proj": torch.nn.Linear(4, 8),
                "down_proj": torch.nn.Linear(8, 4),
            }
        )


def test_linear_module_names_are_collected_from_nested_layers() -> None:
    names = collect_linear_module_names(TinyDecoder())

    assert names == ["mlp.down_proj", "mlp.up_proj", "q_proj"]


def test_target_matching_accepts_suffix_names() -> None:
    matches = target_module_matches(
        ["layers.0.self_attn.q_proj", "layers.0.mlp.down_proj"],
        ["q_proj", "down_proj", "gate_proj"],
    )

    assert matches["q_proj"] == ["layers.0.self_attn.q_proj"]
    assert matches["down_proj"] == ["layers.0.mlp.down_proj"]
    assert matches["gate_proj"] == []


def test_validation_rejects_completely_missing_targets() -> None:
    with pytest.raises(ValueError, match="none of the requested"):
        validate_target_modules(TinyDecoder(), ["not_a_projection"])
