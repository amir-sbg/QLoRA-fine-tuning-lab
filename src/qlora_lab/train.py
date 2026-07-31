from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch

from .config import QLoRAConfig, prepare_output_directories, save_json
from .data import (
    SupervisedDataCollator,
    prepare_datasets,
    tokenized_dataset_profile,
)
from .model import load_qlora_model, load_tokenizer, trainable_parameter_summary


def _training_arguments(config: QLoRAConfig, has_eval: bool):
    from transformers import TrainingArguments

    kwargs = {
        "output_dir": str(config.output_dir),
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "save_total_limit": 2,
        "report_to": "none",
        "optim": "paged_adamw_32bit",
        "fp16": config.compute_dtype == "float16",
        "bf16": config.compute_dtype == "bfloat16" and torch.cuda.is_available(),
        "gradient_checkpointing": True,
        "remove_unused_columns": False,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    eval_key = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    kwargs[eval_key] = "steps" if has_eval else "no"
    if has_eval:
        kwargs["eval_steps"] = config.eval_steps
        kwargs["load_best_model_at_end"] = False
    return TrainingArguments(**kwargs)


def run_training(config: QLoRAConfig) -> dict:
    from transformers import Trainer, set_seed

    prepare_output_directories(config)
    set_seed(config.seed)
    tokenizer = load_tokenizer(config)
    datasets = prepare_datasets(config, tokenizer)
    model = load_qlora_model(config)
    data_profile = {
        "train": tokenized_dataset_profile(datasets["train"]),
        "eval": tokenized_dataset_profile(datasets["eval"]),
    }
    save_json(data_profile, config.report_dir / "data_profile.json")

    collator = SupervisedDataCollator(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=_training_arguments(config, has_eval=len(datasets["eval"]) > 0),
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        data_collator=collator,
        tokenizer=tokenizer,
    )
    result = trainer.train(
        resume_from_checkpoint=(
            str(config.resume_from_checkpoint)
            if config.resume_from_checkpoint is not None
            else None
        )
    )
    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(config.output_dir)

    summary = {
        "config": config.to_dict(),
        "data_profile": data_profile,
        "train_metrics": result.metrics,
        "parameter_summary": trainable_parameter_summary(model),
    }
    save_json(summary, config.report_dir / "train_summary.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune an LLM with QLoRA.")
    parser.add_argument("--model-name", default=QLoRAConfig.model_name)
    parser.add_argument("--dataset-name", default=QLoRAConfig.dataset_name)
    parser.add_argument("--dataset-config")
    parser.add_argument("--dataset-split", default=QLoRAConfig.dataset_split)
    parser.add_argument("--output-dir", type=Path, default=QLoRAConfig.output_dir)
    parser.add_argument("--report-dir", type=Path, default=QLoRAConfig.report_dir)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--max-seq-length", type=int, default=QLoRAConfig.max_seq_length)
    parser.add_argument("--max-train-samples", type=int, default=QLoRAConfig.max_train_samples)
    parser.add_argument("--max-eval-samples", type=int, default=QLoRAConfig.max_eval_samples)
    parser.add_argument("--eval-size", type=float, default=QLoRAConfig.eval_size)
    parser.add_argument("--lora-r", type=int, default=QLoRAConfig.lora_r)
    parser.add_argument("--lora-alpha", type=int, default=QLoRAConfig.lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=QLoRAConfig.lora_dropout)
    parser.add_argument("--target-modules", nargs="+", default=list(QLoRAConfig.target_modules))
    parser.add_argument("--epochs", type=float, default=QLoRAConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=QLoRAConfig.batch_size)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=QLoRAConfig.gradient_accumulation_steps,
    )
    parser.add_argument("--learning-rate", type=float, default=QLoRAConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=QLoRAConfig.weight_decay)
    parser.add_argument("--warmup-ratio", type=float, default=QLoRAConfig.warmup_ratio)
    parser.add_argument("--logging-steps", type=int, default=QLoRAConfig.logging_steps)
    parser.add_argument("--save-steps", type=int, default=QLoRAConfig.save_steps)
    parser.add_argument("--eval-steps", type=int, default=QLoRAConfig.eval_steps)
    parser.add_argument("--seed", type=int, default=QLoRAConfig.seed)
    parser.add_argument("--quant-type", choices=["nf4", "fp4"], default=QLoRAConfig.quant_type)
    parser.add_argument("--compute-dtype", choices=["float16", "bfloat16", "float32"], default=QLoRAConfig.compute_dtype)
    parser.add_argument("--no-double-quant", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> QLoRAConfig:
    values = vars(args).copy()
    values["double_quant"] = not values.pop("no_double_quant")
    values["target_modules"] = tuple(values["target_modules"])
    return QLoRAConfig(**values)


if __name__ == "__main__":
    run_training(config_from_args(build_parser().parse_args()))
