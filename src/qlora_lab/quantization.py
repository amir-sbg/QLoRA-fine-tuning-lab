from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import torch


_NF4_VALUES = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)


@dataclass(frozen=True)
class NF4Tensor:
    codes: torch.Tensor
    scales: torch.Tensor
    original_shape: tuple[int, ...]
    numel: int
    block_size: int

    @property
    def padded_numel(self) -> int:
        return int(self.codes.numel())


def nf4_codebook(device: torch.device | None = None) -> torch.Tensor:
    return torch.tensor(_NF4_VALUES, dtype=torch.float32, device=device)


def quantize_nf4(values: torch.Tensor, block_size: int = 64) -> NF4Tensor:
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if values.numel() == 0:
        raise ValueError("values must not be empty")

    flat = values.detach().to(torch.float32).reshape(-1)
    pad = (-flat.numel()) % block_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))

    blocks = flat.reshape(-1, block_size)
    scales = blocks.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).eps)
    normalized = blocks / scales[:, None]
    codebook = nf4_codebook(values.device)
    distances = (normalized[..., None] - codebook.view(1, 1, -1)) ** 2
    codes = distances.argmin(dim=-1).to(torch.uint8).reshape(-1)
    return NF4Tensor(
        codes=codes,
        scales=scales,
        original_shape=tuple(values.shape),
        numel=values.numel(),
        block_size=block_size,
    )


def dequantize_nf4(tensor: NF4Tensor) -> torch.Tensor:
    if tensor.codes.numel() % tensor.block_size != 0:
        raise ValueError("codes length must be divisible by block_size")
    codebook = nf4_codebook(tensor.codes.device)
    blocks = codebook[tensor.codes.long()].reshape(-1, tensor.block_size)
    values = blocks * tensor.scales[:, None]
    return values.reshape(-1)[: tensor.numel].reshape(tensor.original_shape)


def estimate_4bit_storage_bytes(
    num_values: int,
    block_size: int = 64,
    scale_bytes: int = 2,
) -> int:
    if num_values < 1:
        raise ValueError("num_values must be positive")
    if block_size < 1 or scale_bytes < 1:
        raise ValueError("block_size and scale_bytes must be positive")
    code_bytes = ceil(num_values / 2)
    scale_count = ceil(num_values / block_size)
    return code_bytes + scale_count * scale_bytes


def nf4_error_report(values: torch.Tensor, block_size: int = 64) -> dict[str, float | int]:
    quantized = quantize_nf4(values, block_size=block_size)
    restored = dequantize_nf4(quantized)
    error = (restored - values.detach().to(torch.float32)).reshape(-1)
    fp16_bytes = values.numel() * 2
    nf4_bytes = estimate_4bit_storage_bytes(values.numel(), block_size=block_size)
    return {
        "num_values": int(values.numel()),
        "block_size": int(block_size),
        "mse": float((error**2).mean().item()),
        "mae": float(error.abs().mean().item()),
        "max_abs_error": float(error.abs().max().item()),
        "fp16_bytes": int(fp16_bytes),
        "nf4_bytes": int(nf4_bytes),
        "compression_ratio_vs_fp16": round(fp16_bytes / nf4_bytes, 4),
    }
