from pathlib import Path

import pytest

from qlora_lab.evaluate import load_prompts


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
