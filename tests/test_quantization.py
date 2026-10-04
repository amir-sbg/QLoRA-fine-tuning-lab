import torch
import pytest
from dataclasses import replace

from qlora_lab.quantization import (
    dequantize_nf4,
    estimate_4bit_storage_bytes,
    nf4_error_report,
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


def test_nf4_error_report_exposes_quality_and_storage() -> None:
    values = torch.linspace(-1.0, 1.0, steps=128)
    report = nf4_error_report(values, block_size=32)

    assert report["num_values"] == 128
    assert report["block_size"] == 32
    assert report["mse"] >= 0
    assert report["nf4_bytes"] < report["fp16_bytes"]
    assert report["compression_ratio_vs_fp16"] > 1
    assert report["relative_l2_error"] >= 0
    assert report["sqnr_db"] > 0
    assert 0 <= report["codebook_edge_fraction"] <= 1
    assert report["max_block_scale"] >= report["mean_block_scale"]


def test_nf4_quantizer_rejects_non_finite_weights() -> None:
    with pytest.raises(ValueError, match="finite"):
        quantize_nf4(torch.tensor([0.0, float("nan")]))


def test_nf4_dequantizer_validates_tensor_metadata() -> None:
    quantized = quantize_nf4(torch.arange(8, dtype=torch.float32), block_size=4)

    with pytest.raises(ValueError, match="scale count"):
        dequantize_nf4(replace(quantized, scales=quantized.scales[:1]))
    with pytest.raises(ValueError, match="between 0 and 15"):
        dequantize_nf4(replace(quantized, codes=torch.full_like(quantized.codes, 16)))
