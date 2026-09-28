"""Unified evaluation CLI for D-DOIT benchmark outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from ddoit import BENCHMARKS
from ddoit.benchmarks.ctrldna import default_data_root


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated sequences for a registered benchmark."
    )
    parser.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--sequence-col", default="sequence")
    parser.add_argument("--oracle-type", default="paired")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--constraint", nargs=2, type=float, default=[0.5, 0.5])
    parser.add_argument("--starting-sequences", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--level", default="hard")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.benchmark != "ctrldna":
        raise SystemExit("Native evaluation is currently implemented for --benchmark ctrldna.")

    import pandas as pd

    from ddoit.benchmarks.ctrldna_evaluation import (
        evaluate_sequence_table,
        resolve_starting_sequences,
    )

    input_df = pd.read_csv(args.input)
    starting_sequences = resolve_starting_sequences(
        task=args.task,
        starting_sequences_path=args.starting_sequences,
        level=args.level,
        data_root=args.data_root,
    )
    artifacts = evaluate_sequence_table(
        input_df,
        task=args.task,
        sequence_col=args.sequence_col,
        oracle_type=args.oracle_type,
        device=args.device,
        batch_size=args.batch_size,
        checkpoint_root=args.checkpoint_root,
        constraint=tuple(args.constraint),
        starting_sequences=starting_sequences,
    )
    artifacts.save(args.output, args.summary_output)
    print(f"Saved detail CSV to {args.output}")
    print(f"Saved summary CSV to {args.summary_output}")


if __name__ == "__main__":
    main(sys.argv[1:])
