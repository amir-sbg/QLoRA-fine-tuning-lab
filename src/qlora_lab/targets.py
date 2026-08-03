from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


def _is_linear_like(module: Any) -> bool:
    class_name = module.__class__.__name__.lower()
    return isinstance(module, torch.nn.Linear) or "linear" in class_name


def collect_linear_module_names(model: Any) -> list[str]:
    names = [
        name
        for name, module in model.named_modules()
        if name and _is_linear_like(module)
    ]
    return sorted(set(names))


def target_module_matches(
    available_modules: Iterable[str],
    targets: Iterable[str],
) -> dict[str, list[str]]:
    available = sorted(set(available_modules))
    matches: dict[str, list[str]] = {}
    for target in targets:
        target = target.strip()
        if not target:
            raise ValueError("target module names must not be empty")
        matches[target] = [
            name
            for name in available
            if name == target or name.endswith(f".{target}")
        ]
    return matches


def summarize_target_modules(model: Any, targets: Iterable[str]) -> dict:
    available = collect_linear_module_names(model)
    matches = target_module_matches(available, targets)
    missing = [target for target, rows in matches.items() if not rows]
    matched = {target: rows for target, rows in matches.items() if rows}
    return {
        "available_linear_modules": available,
        "requested_targets": list(matches),
        "matched_targets": matched,
        "missing_targets": missing,
    }


def validate_target_modules(model: Any, targets: Iterable[str]) -> dict:
    summary = summarize_target_modules(model, targets)
    if not summary["matched_targets"]:
        requested = ", ".join(summary["requested_targets"])
        raise ValueError(
            "none of the requested LoRA target modules were found in the model: "
            f"{requested}"
        )
    return summary
