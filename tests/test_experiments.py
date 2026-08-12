import pytest

from qlora_lab.experiments import (
    decoder_block_shapes,
    lora_parameter_count,
    rank_sweep_report,
    save_rank_sweep_csv,
)


def test_lora_parameter_count_uses_input_and_output_dimensions() -> None:
    shapes = decoder_block_shapes(hidden_size=4, intermediate_size=8, layers=1)

    assert len(shapes) == 7
    assert lora_parameter_count(shapes, rank=2) == 2 * (
        4 * (4 + 4) + 3 * (4 + 8)
    )


def test_rank_sweep_grows_linearly_with_rank() -> None:
    report = rank_sweep_report(
        ranks=[4, 8],
        hidden_size=16,
        intermediate_size=32,
        layers=2,
        base_parameters=10_000,
    )
    rows = report["rank_sweep"]

    assert rows[1]["adapter_parameters"] == rows[0]["adapter_parameters"] * 2
    assert rows[0]["scale"] == 2.0


def test_rank_sweep_marks_ranks_under_memory_budget() -> None:
    report = rank_sweep_report(
        ranks=[4, 8, 16],
        hidden_size=64,
        intermediate_size=128,
        layers=4,
        base_parameters=1_000_000,
        adapter_memory_budget_mb=0.05,
    )
    rows = report["rank_sweep"]

    assert any(row["fits_adapter_budget"] for row in rows)
    assert rows[-1]["fits_adapter_budget"] is False
    assert report["assumptions"]["largest_rank_under_budget"] in {4, 8}


def test_rank_sweep_can_be_saved_as_csv(tmp_path) -> None:
    report = rank_sweep_report(
        ranks=[4],
        hidden_size=16,
        intermediate_size=32,
        layers=2,
        base_parameters=10_000,
    )
    output = tmp_path / "rank_sweep.csv"

    save_rank_sweep_csv(report, output)

    rows = output.read_text().splitlines()
    assert rows[0].startswith("hidden_size,intermediate_size,layers")
    assert ",4,8,2.0," in rows[1]


def test_rank_sweep_rejects_empty_rank_list() -> None:
    with pytest.raises(ValueError, match="ranks"):
        rank_sweep_report(
            ranks=[],
            hidden_size=16,
            intermediate_size=32,
            layers=2,
            base_parameters=10_000,
        )


def test_rank_sweep_rejects_bad_base_size() -> None:
    with pytest.raises(ValueError, match="base_parameters"):
        rank_sweep_report(
            ranks=[4],
            hidden_size=16,
            intermediate_size=32,
            layers=2,
            base_parameters=0,
        )
