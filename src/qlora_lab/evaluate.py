from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch

from .config import QLoRAConfig, save_json
from .model import load_tokenizer

WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def load_prompts(prompt: str | None, prompt_file: Path | None) -> list[str]:
    prompts: list[str] = []
    if prompt:
        prompts.append(prompt.strip())
    if prompt_file is not None:
        if not prompt_file.exists():
            raise FileNotFoundError(f"prompt file not found: {prompt_file}")
        prompts.extend(
            line.strip()
            for line in prompt_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    prompts = [value for value in prompts if value]
    if not prompts:
        raise ValueError("provide --prompt, --prompt-file, or both")
    return prompts


def build_generation_kwargs(
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> dict:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    return kwargs


def normalized_words(text: str) -> list[str]:
    return [match.group(0).casefold() for match in WORD_RE.finditer(text)]


def repeated_bigram_rate(text: str) -> float:
    words = normalized_words(text)
    if len(words) < 3:
        return 0.0
    bigrams = list(zip(words, words[1:]))
    repeated = len(bigrams) - len(set(bigrams))
    return round(repeated / len(bigrams), 4)


def generation_review_row(result: dict) -> dict[str, float | int | bool]:
    prompt_words = normalized_words(str(result.get("prompt", "")))
    generation_words = normalized_words(str(result.get("generation", "")))
    unique_ratio = (
        len(set(generation_words)) / len(generation_words)
        if generation_words
        else 0.0
    )
    return {
        "prompt_words": len(prompt_words),
        "generation_words": len(generation_words),
        "generation_chars": len(str(result.get("generation", ""))),
        "empty_generation": not bool(generation_words),
        "unique_word_ratio": round(unique_ratio, 4),
        "repeated_bigram_rate": repeated_bigram_rate(
            str(result.get("generation", "")),
        ),
    }


def summarize_generation_review(results: list[dict]) -> dict:
    rows = [generation_review_row(result) for result in results]
    if not rows:
        return {
            "examples": 0,
            "empty_generations": 0,
            "avg_prompt_words": 0.0,
            "avg_generation_words": 0.0,
            "avg_unique_word_ratio": 0.0,
            "max_repeated_bigram_rate": 0.0,
            "rows": [],
        }

    def avg(key: str) -> float:
        return round(sum(float(row[key]) for row in rows) / len(rows), 3)

    return {
        "examples": len(rows),
        "empty_generations": sum(1 for row in rows if row["empty_generation"]),
        "avg_prompt_words": avg("prompt_words"),
        "avg_generation_words": avg("generation_words"),
        "avg_unique_word_ratio": avg("unique_word_ratio"),
        "max_repeated_bigram_rate": max(
            float(row["repeated_bigram_rate"]) for row in rows
        ),
        "rows": rows,
    }


def save_generation_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model_name", "adapter_dir", "prompt", "generation"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key, "") for key in fieldnames})


def generate_from_adapter(
    config: QLoRAConfig,
    adapter_dir: Path,
    prompt: str,
    max_new_tokens: int = 120,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> dict:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(config)
    base = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            **build_generation_kwargs(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            ),
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {
        "model_name": config.model_name,
        "adapter_dir": str(adapter_dir),
        "prompt": prompt,
        "generation": text[len(prompt) :].strip() if text.startswith(prompt) else text,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text from a saved LoRA adapter.")
    parser.add_argument("--model-name", default=QLoRAConfig.model_name)
    parser.add_argument("--adapter-dir", type=Path, default=QLoRAConfig.output_dir)
    parser.add_argument("--report-dir", type=Path, default=QLoRAConfig.report_dir)
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--csv-output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = QLoRAConfig(model_name=args.model_name, report_dir=args.report_dir)
    results = [
        generate_from_adapter(
            config=config,
            adapter_dir=args.adapter_dir,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        for prompt in load_prompts(args.prompt, args.prompt_file)
    ]
    save_json(
        {
            "generations": results,
            "generation_review": summarize_generation_review(results),
        },
        config.report_dir / "generations.json",
    )
    if args.csv_output is not None:
        save_generation_csv(results, args.csv_output)
    for index, result in enumerate(results, start=1):
        print(f"[{index}] {result['generation']}")


if __name__ == "__main__":
    main()
