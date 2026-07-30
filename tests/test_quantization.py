import torch

from qlora_lab.quantization import (
    dequantize_nf4,
    estimate_4bit_storage_bytes,
    nf4_codebook,
    quantize_nf4,
)


def test_nf4_codebook_has_sixteen_ordered_values() -> None:
    codebook = nf4_codebook()

    assert codebook.shape == (16,)
    assert torch.all(codebook[1:] >= codebook[:-1])
    assert codebook[0].item() == -1.0
    assert codebook[-1].item() == 1.0


def test_nf4_round_trip_preserves_shape_and_range() -> None:
    values = torch.linspace(-2.0, 2.0, steps=31).reshape(31, 1)
    quantized = quantize_nf4(values, block_size=8)
    restored = dequantize_nf4(quantized)

    assert restored.shape == values.shape
    assert quantized.codes.max().item() <= 15
    assert torch.max(torch.abs(restored - values)).item() < 0.35


def test_storage_estimate_accounts_for_codes_and_scales() -> None:
    assert estimate_4bit_storage_bytes(64, block_size=64, scale_bytes=2) == 34
