from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QLoRAConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_name: str = "databricks/databricks-dolly-15k"
    dataset_config: str | None = None
    dataset_split: str = "train"
    output_dir: Path = Path("artifacts/qlora-adapter")
    report_dir: Path = Path("reports")
    resume_from_checkpoint: Path | None = None

    max_seq_length: int = 512
    max_train_samples: int | None = 2000
    max_eval_samples: int | None = 200
    eval_size: float = 0.10

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    epochs: float = 1.0
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    seed: int = 42

    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model_name must not be empty")
        if not self.dataset_name:
            raise ValueError("dataset_name must not be empty")
        if self.max_seq_length < 32:
            raise ValueError("max_seq_length must be at least 32")
        if self.max_train_samples is not None and self.max_train_samples < 1:
            raise ValueError("max_train_samples must be positive or None")
        if self.max_eval_samples is not None and self.max_eval_samples < 1:
            raise ValueError("max_eval_samples must be positive or None")
        if not 0.0 < self.eval_size < 0.5:
            raise ValueError("eval_size must be between 0 and 0.5")
        if self.lora_r < 1 or self.lora_alpha < 1:
            raise ValueError("lora_r and lora_alpha must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")
        if not self.target_modules:
            raise ValueError("target_modules must not be empty")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("batch settings must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer settings are invalid")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.logging_steps < 1 or self.save_steps < 1 or self.eval_steps < 1:
            raise ValueError("logging, save, and eval steps must be positive")
        if self.quant_type not in {"nf4", "fp4"}:
            raise ValueError("quant_type must be 'nf4' or 'fp4'")
        if self.compute_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("compute_dtype must be float16, bfloat16, or float32")

    @property
    def lora_scale(self) -> float:
        return self.lora_alpha / self.lora_r

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["report_dir"] = str(self.report_dir)
        payload["resume_from_checkpoint"] = (
            str(self.resume_from_checkpoint)
            if self.resume_from_checkpoint is not None
            else None
        )
        payload["target_modules"] = list(self.target_modules)
        payload["lora_scale"] = self.lora_scale
        return payload


def prepare_output_directories(config: QLoRAConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
