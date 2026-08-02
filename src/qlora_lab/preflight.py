from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch

from .config import QLoRAConfig, save_json
from .quantization import estimate_4bit_storage_bytes


def effective_batch_size(config: QLoRAConfig, world_size: int = 1) -> int:
    if world_size < 1:
        raise ValueError("world_size must be positive")
    return config.batch_size * config.gradient_accumulation_steps * world_size


def estimate_update_steps(
    train_examples: int,
    config: QLoRAConfig,
    world_size: int = 1,
) -> dict[str, int]:
    if train_examples < 1:
        raise ValueError("train_examples must be positive")
    batch = effective_batch_size(config, world_size)
    steps_per_epoch = math.ceil(train_examples / batch)
    total_steps = max(1, math.ceil(steps_per_epoch * config.epochs))
    return {
        "train_examples": train_examples,
        "effective_batch_size": batch,
        "steps_per_epoch": steps_per_epoch,
        "estimated_total_steps": total_steps,
    }


def runtime_summary() -> dict[str, Any]:
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_gb": round(properties.total_memory / 1_073_741_824, 3),
                }
            )
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "bf16_supported": bool(
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        ),
        "devices": devices,
    }


def preflight_warnings(config: QLoRAConfig, runtime: dict[str, Any]) -> list[str]:
    warnings = []
    if not runtime["cuda_available"]:
        warnings.append("QLoRA training expects CUDA because bitsandbytes 4-bit kernels run on NVIDIA GPUs.")
    if config.compute_dtype == "bfloat16" and not runtime["bf16_supported"]:
        warnings.append("bfloat16 was requested but the current CUDA runtime does not report bf16 support.")
    if config.max_seq_length > 2048 and config.batch_size > 1:
        warnings.append("Large sequence length with batch_size > 1 can make activation memory the bottleneck.")
    if config.lora_alpha / config.lora_r > 4:
        warnings.append("LoRA scale is high; watch early training loss for unstable updates.")
    return warnings


def build_preflight_report(
    config: QLoRAConfig,
    train_examples: int | None = None,
    base_parameters: int | None = None,
    world_size: int = 1,
) -> dict[str, Any]:
    runtime = runtime_summary()
    report: dict[str, Any] = {
        "model_name": config.model_name,
        "dataset_name": config.dataset_name,
        "runtime": runtime,
        "batching": {
            "per_device_batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "world_size": world_size,
            "effective_batch_size": effective_batch_size(config, world_size),
        },
        "qlora": {
            "quant_type": config.quant_type,
            "double_quant": config.double_quant,
            "compute_dtype": config.compute_dtype,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_scale": config.lora_scale,
            "max_grad_norm": config.max_grad_norm,
            "target_modules": list(config.target_modules),
        },
        "warnings": preflight_warnings(config, runtime),
    }
    if train_examples is not None:
        report["steps"] = estimate_update_steps(train_examples, config, world_size)
    if base_parameters is not None:
        report["memory_estimate"] = {
            "base_parameters": base_parameters,
            "base_4bit_memory_mb": round(
                estimate_4bit_storage_bytes(base_parameters) / 1_048_576,
                3,
            ),
        }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run QLoRA training preflight checks.")
    parser.add_argument("--model-name", default=QLoRAConfig.model_name)
    parser.add_argument("--dataset-name", default=QLoRAConfig.dataset_name)
    parser.add_argument("--max-seq-length", type=int, default=QLoRAConfig.max_seq_length)
    parser.add_argument("--lora-r", type=int, default=QLoRAConfig.lora_r)
    parser.add_argument("--lora-alpha", type=int, default=QLoRAConfig.lora_alpha)
    parser.add_argument("--batch-size", type=int, default=QLoRAConfig.batch_size)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=QLoRAConfig.gradient_accumulation_steps,
    )
    parser.add_argument("--epochs", type=float, default=QLoRAConfig.epochs)
    parser.add_argument("--compute-dtype", choices=["float16", "bfloat16", "float32"], default=QLoRAConfig.compute_dtype)
    parser.add_argument("--quant-type", choices=["nf4", "fp4"], default=QLoRAConfig.quant_type)
    parser.add_argument("--max-grad-norm", type=float, default=QLoRAConfig.max_grad_norm)
    parser.add_argument("--train-examples", type=int)
    parser.add_argument("--base-parameters", type=int)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("reports/preflight.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = QLoRAConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        compute_dtype=args.compute_dtype,
        quant_type=args.quant_type,
        max_grad_norm=args.max_grad_norm,
    )
    report = build_preflight_report(
        config,
        train_examples=args.train_examples,
        base_parameters=args.base_parameters,
        world_size=args.world_size,
    )
    save_json(report, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
