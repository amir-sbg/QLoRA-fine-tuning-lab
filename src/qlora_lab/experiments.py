from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import save_json
from .quantization import estimate_4bit_storage_bytes


@dataclass(frozen=True)
class LinearShape:
    name: str
    in_features: int
    out_features: int


def decoder_block_shapes(
    hidden_size: int,
    intermediate_size: int,
    layers: int,
) -> list[LinearShape]:
    if hidden_size < 1 or intermediate_size < 1 or layers < 1:
        raise ValueError("model dimensions must be positive")

    block = [
        LinearShape("q_proj", hidden_size, hidden_size),
        LinearShape("k_proj", hidden_size, hidden_size),
        LinearShape("v_proj", hidden_size, hidden_size),
        LinearShape("o_proj", hidden_size, hidden_size),
        LinearShape("gate_proj", hidden_size, intermediate_size),
        LinearShape("up_proj", hidden_size, intermediate_size),
        LinearShape("down_proj", intermediate_size, hidden_size),
    ]
    return [
        LinearShape(f"layer_{layer}.{shape.name}", shape.in_features, shape.out_features)
        for layer in range(layers)
        for shape in block
    ]


def lora_parameter_count(shapes: Iterable[LinearShape], rank: int) -> int:
    if rank < 1:
        raise ValueError("rank must be positive")
    return sum(rank * (shape.in_features + shape.out_features) for shape in shapes)


def rank_sweep_report(
    ranks: Iterable[int],
    hidden_size: int,
    intermediate_size: int,
    layers: int,
    base_parameters: int,
    alpha_multiplier: int = 2,
) -> dict:
    shapes = decoder_block_shapes(hidden_size, intermediate_size, layers)
    base_4bit_bytes = estimate_4bit_storage_bytes(base_parameters)
    rows = []
    for rank in ranks:
        alpha = rank * alpha_multiplier
        adapter_parameters = lora_parameter_count(shapes, rank)
        rows.append(
            {
                "rank": rank,
                "alpha": alpha,
                "scale": alpha / rank,
                "adapter_parameters": adapter_parameters,
                "adapter_memory_mb_fp16": round(adapter_parameters * 2 / 1_048_576, 3),
                "adapter_vs_base_percent": round(
                    100 * adapter_parameters / base_parameters,
                    4,
                ),
            }
        )
    return {
        "assumptions": {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "layers": layers,
            "base_parameters": base_parameters,
            "base_4bit_memory_mb": round(base_4bit_bytes / 1_048_576, 3),
            "target_modules": sorted({shape.name.split(".")[-1] for shape in shapes}),
        },
        "rank_sweep": rows,
    }


def rank_sweep_csv_rows(report: dict) -> list[dict]:
    assumptions = report["assumptions"]
    context = {
        "hidden_size": assumptions["hidden_size"],
        "intermediate_size": assumptions["intermediate_size"],
        "layers": assumptions["layers"],
        "base_parameters": assumptions["base_parameters"],
        "base_4bit_memory_mb": assumptions["base_4bit_memory_mb"],
    }
    return [{**context, **row} for row in report["rank_sweep"]]


def save_rank_sweep_csv(report: dict, path: Path) -> None:
    rows = rank_sweep_csv_rows(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate QLoRA rank tradeoffs.")
    parser.add_argument("--ranks", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=8192)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--base-parameters", type=int, default=1_500_000_000)
    parser.add_argument("--output", type=Path, default=Path("reports/rank_sweep.json"))
    parser.add_argument("--csv-output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = rank_sweep_report(
        ranks=args.ranks,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        layers=args.layers,
        base_parameters=args.base_parameters,
    )
    save_json(report, args.output)
    print(f"wrote {args.output}")
    if args.csv_output is not None:
        save_rank_sweep_csv(report, args.csv_output)
        print(f"wrote {args.csv_output}")


if __name__ == "__main__":
    main()
