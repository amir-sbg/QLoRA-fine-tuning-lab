from __future__ import annotations

from math import acos, degrees, sqrt

import torch


def lora_scaling(alpha: float, rank: int, use_rslora: bool = False) -> float:
    if rank < 1 or alpha <= 0:
        raise ValueError("rank and alpha must be positive")
    return float(alpha / (sqrt(rank) if use_rslora else rank))


def merge_lora_weight(
    base_weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    alpha: float,
    use_rslora: bool = False,
) -> torch.Tensor:
    if base_weight.ndim != 2 or lora_a.ndim != 2 or lora_b.ndim != 2:
        raise ValueError("base weight and LoRA factors must be 2-D matrices")
    if lora_a.shape[0] != lora_b.shape[1]:
        raise ValueError("LoRA factor rank dimensions must match")
    if base_weight.shape != (lora_b.shape[0], lora_a.shape[1]):
        raise ValueError("base_weight must match the LoRA update shape")
    if not all(torch.isfinite(tensor).all() for tensor in (base_weight, lora_a, lora_b)):
        raise ValueError("weights must contain only finite values")
    scale = lora_scaling(alpha, lora_a.shape[0], use_rslora)
    update = lora_b.to(torch.float32) @ lora_a.to(torch.float32)
    return base_weight.to(torch.float32) + scale * update


def lora_update_diagnostics(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    alpha: float,
    base_weight: torch.Tensor | None = None,
    use_rslora: bool = False,
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
    scale = lora_scaling(alpha, rank, use_rslora)
    update = scale * (lora_b.float() @ lora_a.float())
    frobenius_norm = float(torch.linalg.matrix_norm(update, ord="fro"))
    spectral_norm = float(torch.linalg.matrix_norm(update, ord=2))
    stable_rank = frobenius_norm**2 / max(spectral_norm**2, 1e-24)
    report: dict[str, float | int] = {
        "rank": int(rank),
        "alpha": float(alpha),
        "scale": float(scale),
        "scaling_rule": "alpha/sqrt(r)" if use_rslora else "alpha/r",
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
        flat_base = base_weight.float().reshape(-1)
        flat_update = update.reshape(-1)
        cosine = float(
            torch.dot(flat_base, flat_update)
            / (torch.linalg.vector_norm(flat_base) * torch.linalg.vector_norm(flat_update)).clamp_min(1e-12)
        )
        cosine = max(-1.0, min(1.0, cosine))
        report["base_frobenius_norm"] = base_norm
        report["relative_update_norm"] = frobenius_norm / max(base_norm, 1e-12)
        report["base_update_cosine"] = cosine
        report["base_update_angle_degrees"] = degrees(acos(cosine))
        active = (flat_base != 0) & (flat_update != 0)
        report["base_update_sign_agreement"] = (
            float((torch.sign(flat_base[active]) == torch.sign(flat_update[active])).float().mean())
            if torch.any(active)
            else 0.0
        )
    return report


def lora_spectrum_report(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    alpha: float,
    use_rslora: bool = False,
) -> dict[str, float | int]:
    if lora_a.ndim != 2 or lora_b.ndim != 2:
        raise ValueError("LoRA factors must be 2-D matrices")
    if lora_a.shape[0] != lora_b.shape[1]:
        raise ValueError("LoRA factor rank dimensions must match")
    if not torch.isfinite(lora_a).all() or not torch.isfinite(lora_b).all():
        raise ValueError("LoRA factors must contain only finite values")

    update = lora_scaling(alpha, lora_a.shape[0], use_rslora) * (
        lora_b.float() @ lora_a.float()
    )
    singular_values = torch.linalg.svdvals(update)
    energy = singular_values.square()
    fractions = energy / energy.sum().clamp_min(1e-24)
    nonzero = fractions > 0
    entropy = -torch.sum(fractions[nonzero] * torch.log(fractions[nonzero]))
    cumulative = torch.cumsum(fractions, dim=0)
    rank_90 = int(torch.searchsorted(cumulative, 0.90).item() + 1)
    tolerance = singular_values[0] * max(update.shape) * torch.finfo(singular_values.dtype).eps
    active_singular_values = singular_values[singular_values > tolerance]
    active_rank = int(active_singular_values.numel())
    participation_ratio = float(1.0 / torch.sum(fractions.square()).clamp_min(1e-24))
    return {
        "top_singular_value": float(singular_values[0]),
        "top1_energy_fraction": float(fractions[0]),
        "rank_90_percent_energy": rank_90,
        "entropy_effective_rank": float(torch.exp(entropy)),
        "energy_participation_rank": participation_ratio,
        "active_rank": active_rank,
        "configured_rank": int(lora_a.shape[0]),
        "active_rank_fraction": active_rank / lora_a.shape[0],
        "active_condition_number": float(
            active_singular_values[0] / active_singular_values[-1]
        ),
    }
