from pathlib import Path

import pytest

from qlora_lab.evaluate import (
    build_generation_kwargs,
    load_prompts,
    save_generation_csv,
)


def test_load_prompts_accepts_prompt_and_file(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text(
        "# skipped comment\n\nExplain LoRA.\nSummarize NF4.\n"
    )

    prompts = load_prompts("Explain QLoRA.", prompt_file)

    assert prompts == ["Explain QLoRA.", "Explain LoRA.", "Summarize NF4."]


def test_load_prompts_requires_input() -> None:
    with pytest.raises(ValueError, match="provide --prompt"):
        load_prompts(None, None)


def test_generation_kwargs_keep_deterministic_default() -> None:
    kwargs = build_generation_kwargs(max_new_tokens=32)

    assert kwargs == {"max_new_tokens": 32, "do_sample": False}


def test_generation_kwargs_include_sampling_options() -> None:
    kwargs = build_generation_kwargs(
        max_new_tokens=32,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
    )

    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 0.8
    assert kwargs["top_p"] == 0.9


def test_generation_kwargs_reject_invalid_top_p() -> None:
    with pytest.raises(ValueError, match="top_p"):
        build_generation_kwargs(max_new_tokens=32, top_p=1.5)


def test_generation_results_can_be_saved_as_csv(tmp_path: Path) -> None:
    output = tmp_path / "generations.csv"

    save_generation_csv(
        [
            {
                "model_name": "tiny",
                "adapter_dir": "adapter",
                "prompt": "Explain LoRA.",
                "generation": "Low-rank adapters.",
            }
        ],
        output,
    )

    assert output.read_text().splitlines() == [
        "model_name,adapter_dir,prompt,generation",
        "tiny,adapter,Explain LoRA.,Low-rank adapters.",
    ]
