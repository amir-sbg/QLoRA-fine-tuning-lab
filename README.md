# QLoRA Fine-Tuning Lab

A compact fine-tuning project for adapting causal language models with QLoRA. The code covers dataset formatting, prompt/label masking, 4-bit model loading, LoRA adapter training, lightweight generation evaluation, and small rank experiments that make the parameter and memory tradeoffs visible.

## Why QLoRA

QLoRA keeps the pretrained backbone frozen in 4-bit form and trains small low-rank adapter matrices. For a linear layer, the effective weight is:

```text
W_eff = dequant_4bit(W_q) + (alpha / r) * B @ A
```

`W_q` is the quantized base weight, while `A` and `B` are the trainable LoRA matrices. Increasing rank `r` gives the adapter more capacity, but it also increases memory and optimizer state. This project keeps those choices explicit in the config and in the experiment report.

The training path uses the normal Hugging Face stack: `transformers`, `datasets`, `peft`, `accelerate`, and `bitsandbytes`. The `qlora_lab.quantization` module includes a small NF4 block quantizer as a reference implementation so the 4-bit idea is testable without needing a GPU.

## Project Flow

1. Load an instruction dataset such as `databricks/databricks-dolly-15k`.
2. Format each row into an instruction, optional context, and response block.
3. Tokenize the full sequence while masking the prompt tokens from the loss.
4. Load the base model in 4-bit NF4 with optional double quantization.
5. Check the requested adapter target modules against the loaded model, then attach LoRA adapters to the matched projection layers.
6. Train only the adapter weights and save the adapter checkpoint.
7. Save a token profile so sequence length and supervised-token ratio are easy to inspect.
8. Run a preflight report before using GPU time.
9. Run a rank sweep report to compare adapter size, scale, and memory.

## Setup

QLoRA training needs an NVIDIA GPU with CUDA-compatible `bitsandbytes`. The tests and rank-sweep report run on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

## Run a Small Training Job

```bash
python -m qlora_lab.train \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --dataset-name databricks/databricks-dolly-15k \
  --max-train-samples 1000 \
  --max-eval-samples 200 \
  --max-seq-length 512 \
  --lora-r 16 \
  --lora-alpha 32 \
  --epochs 1 \
  --batch-size 2 \
  --gradient-accumulation-steps 8
```

Adapters are saved under `artifacts/qlora-adapter/`. Training metadata is written to `reports/train_summary.json`, and token statistics are written to `reports/data_profile.json`.

Interrupted runs can be resumed from a Trainer checkpoint:

```bash
python -m qlora_lab.train \
  --resume-from-checkpoint artifacts/qlora-adapter/checkpoint-100
```

## Preflight Check

```bash
python -m qlora_lab.preflight \
  --train-examples 1000 \
  --base-parameters 500000000 \
  --batch-size 2 \
  --gradient-accumulation-steps 8
```

The report includes CUDA availability, effective batch size, estimated update steps, 4-bit base-weight memory, LoRA scale, and basic warnings that are useful before starting a run.

## Rank Experiment

```bash
python -m qlora_lab.experiments \
  --ranks 4 8 16 32 \
  --hidden-size 2048 \
  --intermediate-size 8192 \
  --layers 24 \
  --base-parameters 1500000000 \
  --output reports/rank_sweep.json
```

This writes a JSON report with LoRA parameter counts, FP16 adapter memory, estimated 4-bit backbone memory, and the `alpha / r` scaling used by the adapter update.

## Evaluate an Adapter

```bash
python -m qlora_lab.evaluate \
  --adapter-dir artifacts/qlora-adapter \
  --prompt "Explain why gradient accumulation is useful for QLoRA."
```

For a small repeatable prompt set, put one prompt per line and pass `--prompt-file`:

```bash
python -m qlora_lab.evaluate \
  --adapter-dir artifacts/qlora-adapter \
  --prompt-file prompts/eval_prompts.txt
```

The evaluator loads the saved adapter and writes generated samples to `reports/generations.json`.

## Repository Layout

```text
src/qlora_lab/
├── config.py        # shared training configuration and validation
├── data.py          # instruction formatting and label masking
├── model.py         # 4-bit base model and PEFT adapter setup
├── quantization.py  # NF4 reference implementation
├── targets.py       # adapter target-module inspection
├── preflight.py     # runtime and training sanity checks
├── experiments.py   # rank and memory comparison report
├── train.py         # command-line training pipeline
└── evaluate.py      # adapter generation script
```

## Suggested Repo Name

`QLoRA Fine-Tuning Lab`

## About

Reproducible QLoRA fine-tuning pipeline for 4-bit LLM adaptation, with NF4 quantization math, LoRA rank experiments, prompt masking, adapter training, and lightweight generation evaluation.
