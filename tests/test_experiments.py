from qlora_lab.experiments import (
    decoder_block_shapes,
    lora_parameter_count,
    rank_sweep_report,
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
