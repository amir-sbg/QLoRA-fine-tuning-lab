from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import QLoRAConfig, save_json
from .model import load_tokenizer


def load_prompts(prompt: str | None, prompt_file: Path | None) -> list[str]:
    prompts: list[str] = []
    if prompt:
        prompts.append(prompt.strip())
    if prompt_file is not None:
        if not prompt_file.exists():
            raise FileNotFoundError(f"prompt file not found: {prompt_file}")
        prompts.extend(
            line.strip()
            for line in prompt_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    prompts = [value for value in prompts if value]
    if not prompts:
        raise ValueError("provide --prompt, --prompt-file, or both")
    return prompts


def generate_from_adapter(
    config: QLoRAConfig,
    adapter_dir: Path,
    prompt: str,
    max_new_tokens: int = 120,
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
            max_new_tokens=max_new_tokens,
            do_sample=False,
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
        )
        for prompt in load_prompts(args.prompt, args.prompt_file)
    ]
    save_json({"generations": results}, config.report_dir / "generations.json")
    for index, result in enumerate(results, start=1):
        print(f"[{index}] {result['generation']}")


if __name__ == "__main__":
    main()
