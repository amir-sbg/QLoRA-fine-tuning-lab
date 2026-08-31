from pathlib import Path

import pytest

from qlora_lab.evaluate import (
    build_generation_kwargs,
    load_prompts,
    repeated_bigram_rate,
    save_generation_csv,
    summarize_generation_review,
)


def test_load_prompts_accepts_prompt_and_file(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text(
        "# skipped comment\n\nExplain LoRA.\nSummarize NF4.\nچرا QLoRA مفید است؟\n",
        encoding="utf-8",
    )

    prompts = load_prompts("Explain QLoRA.", prompt_file)

    assert prompts == [
        "Explain QLoRA.",
        "Explain LoRA.",
        "Summarize NF4.",
        "چرا QLoRA مفید است؟",
    ]


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


def test_repeated_bigram_rate_flags_looping_text() -> None:
    assert repeated_bigram_rate("adapter learns adapter learns adapter learns") > 0
    assert repeated_bigram_rate("adapter learns a compact residual update") == 0.0


def test_generation_review_summarizes_outputs() -> None:
    report = summarize_generation_review(
        [
            {"prompt": "Explain LoRA", "generation": "Low rank adapter update."},
            {"prompt": "Explain NF4", "generation": ""},
        ]
    )

    assert report["examples"] == 2
    assert report["empty_generations"] == 1
    assert report["avg_prompt_words"] == 2.0
    assert report["avg_generation_words"] == 2.0
    assert report["rows"][0]["unique_word_ratio"] == 1.0
