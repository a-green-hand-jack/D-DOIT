"""Unified generation CLI for D-DOIT benchmarks.

This module is a compatibility dispatcher during the codebase migration. It
normalizes benchmark and method names, then delegates to the legacy runnable
scripts until their implementation has been moved under ``ddoit``.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from ddoit import BENCHMARKS, get_benchmark
from ddoit.benchmarks.ctrldna import build_legacy_generate_command
from ddoit.benchmarks.drakes import build_legacy_generate_command as build_drakes_command
from ddoit.core import CommandPlan


REPO_ROOT = Path(__file__).resolve().parents[2]

def _build_ctrldna_command(args: argparse.Namespace, extra_args: Sequence[str]) -> CommandPlan:
    try:
        return build_legacy_generate_command(args, extra_args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def build_command(args: argparse.Namespace, extra_args: Sequence[str]) -> CommandPlan:
    """Build a benchmark-specific legacy command from unified CLI args."""

    if args.benchmark == "drakes":
        try:
            return build_drakes_command(args, extra_args)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    if args.benchmark == "ctrldna":
        return _build_ctrldna_command(args, extra_args)
    valid = ", ".join(sorted(BENCHMARKS))
    raise SystemExit(f"Unknown benchmark '{args.benchmark}'. Valid benchmarks: {valid}")


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Dispatch D-DOIT sequence generation to a registered benchmark."
    )
    parser.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    parser.add_argument("--method", required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument("--num-sequences", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)

    drakes = parser.add_argument_group("DRAKES compatibility")
    drakes.add_argument("--output-dir", type=Path, default=Path("outputs/ddoit/drakes"))
    drakes.add_argument("--num-samples-per-batch", type=int, default=None)
    drakes.add_argument("--ckpt-path", type=Path, default=None)
    drakes.add_argument("--pretrained-path", type=Path, default=None)
    drakes.add_argument("--zero-alpha-path", type=Path, default=None)
    drakes.add_argument("--cfg-path", type=Path, default=None)

    ctrldna = parser.add_argument_group("Ctrl-DNA compatibility")
    ctrldna.add_argument(
        "--legacy-ctrldna-root",
        type=Path,
        default=get_benchmark("ctrldna").legacy_paths[0],
    )
    ctrldna.add_argument("--checkpoint", type=Path, default=None)
    ctrldna.add_argument("--checkpoint-root", type=Path, default=None)
    ctrldna.add_argument("--output", type=Path, default=None)
    ctrldna.add_argument("--summary-output", type=Path, default=None)
    ctrldna.add_argument("--num-steps", type=int, default=None)
    ctrldna.add_argument("--beta", type=float, default=None)
    ctrldna.add_argument("--resample-every", type=int, default=None)
    ctrldna.add_argument("--guidance-scale", type=float, default=None)
    ctrldna.add_argument("--K", type=int, default=None)
    ctrldna.add_argument("--bok-split-step", type=int, default=None)
    ctrldna.add_argument("--doit-M", type=int, default=None)
    ctrldna.add_argument("--doit-omega", type=float, default=None)
    ctrldna.add_argument("--doit-beta", type=float, default=None)
    ctrldna.add_argument("--doit-alpha", type=float, default=None)
    ctrldna.add_argument("--doit-l-star", type=int, default=None)
    ctrldna.add_argument("--doit-gamma", type=float, default=None)
    ctrldna.add_argument("--batch-size", type=int, default=None)
    ctrldna.add_argument("--reward-mode", choices=["weighted", "composite"], default=None)
    ctrldna.add_argument("--reward-lambda", type=float, default=None)
    ctrldna.add_argument("--device", default=None)
    ctrldna.add_argument("--reference-csv", type=Path, default=None)
    ctrldna.add_argument("--reference-col", default=None)
    ctrldna.add_argument("--reference-n", type=int, default=None)

    args, extra_args = parser.parse_known_args(argv)
    if args.num_sequences <= 0:
        parser.error("--num-sequences must be positive")
    if args.num_samples_per_batch is not None and args.num_samples_per_batch <= 0:
        parser.error("--num-samples-per-batch must be positive")
    return args, extra_args


def main(argv: Sequence[str] | None = None) -> None:
    args, extra_args = parse_args(argv)
    plan = build_command(args, extra_args)

    if plan.notes:
        for note in plan.notes:
            print(f"note: {note}", file=sys.stderr)

    if args.dry_run:
        print(f"cwd: {plan.cwd}")
        print(shlex.join(plan.command))
        return

    subprocess.run(plan.command, cwd=plan.cwd, check=True)


if __name__ == "__main__":
    main()
