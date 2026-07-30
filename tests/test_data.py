from qlora_lab.data import build_prompt, extract_instruction_fields, tokenize_example


class ToyTokenizer:
    eos_token = "<eos>"

    def __call__(
        self,
        text,
        max_length,
        truncation,
        add_special_tokens=False,
        return_tensors=None,
    ):
        pieces = text.split()
        if truncation:
            pieces = pieces[:max_length]
        ids = list(range(1, len(pieces) + 1))
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_prompt_includes_context_when_available() -> None:
    prompt = build_prompt("Summarize", "A short article")

    assert "### Instruction:" in prompt
    assert "### Context:" in prompt
    assert prompt.endswith("### Response:\n")


def test_extract_instruction_fields_supports_common_columns() -> None:
    instruction, context, response = extract_instruction_fields(
        {"question": "What is LoRA?", "context": "adapters", "answer": "low rank"}
    )

    assert instruction == "What is LoRA?"
    assert context == "adapters"
    assert response == "low rank"


def test_tokenize_masks_prompt_tokens() -> None:
    encoded = tokenize_example(
        {"instruction": "Say hello", "response": "hello there"},
        ToyTokenizer(),
        max_seq_length=64,
    )

    assert len(encoded["input_ids"]) == len(encoded["labels"])
    assert -100 in encoded["labels"]
    assert encoded["labels"][-1] != -100
