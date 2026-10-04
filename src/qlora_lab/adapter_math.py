from __future__ import annotations

import torch


def lora_update_diagnostics(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    alpha: float,
    base_weight: torch.Tensor | None = None,
) -> dict[str, float | int]:
    if lora_a.ndim != 2 or lora_b.ndim != 2:
        raise ValueError("LoRA factors must be 2-D matrices")
    if lora_a.shape[0] != lora_b.shape[1]:
        raise ValueError("LoRA factor rank dimensions must match")
    if lora_a.shape[0] < 1 or alpha <= 0:
        raise ValueError("rank and alpha must be positive")
    if not torch.isfinite(lora_a).all() or not torch.isfinite(lora_b).all():
        raise ValueError("LoRA factors must contain only finite values")

    rank = lora_a.shape[0]
    scale = alpha / rank
    update = scale * (lora_b.float() @ lora_a.float())
    frobenius_norm = float(torch.linalg.matrix_norm(update, ord="fro"))
    spectral_norm = float(torch.linalg.matrix_norm(update, ord=2))
    stable_rank = frobenius_norm**2 / max(spectral_norm**2, 1e-24)
    report: dict[str, float | int] = {
        "rank": int(rank),
        "alpha": float(alpha),
        "scale": float(scale),
        "a_frobenius_norm": float(torch.linalg.matrix_norm(lora_a.float(), ord="fro")),
        "b_frobenius_norm": float(torch.linalg.matrix_norm(lora_b.float(), ord="fro")),
        "update_frobenius_norm": frobenius_norm,
        "update_spectral_norm": spectral_norm,
        "update_stable_rank": float(stable_rank),
        "update_numerical_rank": int(torch.linalg.matrix_rank(update)),
    }
    if base_weight is not None:
        if base_weight.shape != update.shape:
            raise ValueError("base_weight must match the LoRA update shape")
        if not torch.isfinite(base_weight).all():
            raise ValueError("base_weight must contain only finite values")
        base_norm = float(torch.linalg.matrix_norm(base_weight.float(), ord="fro"))
        report["base_frobenius_norm"] = base_norm
        report["relative_update_norm"] = frobenius_norm / max(base_norm, 1e-12)
    return report
