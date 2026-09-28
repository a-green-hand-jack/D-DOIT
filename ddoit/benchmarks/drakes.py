"""DRAKES/HepG2 benchmark adapter.

This module owns the dependency-free benchmark contract, method normalization,
and command planning used by the unified ``ddoit`` CLI.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from ddoit.core import (
    BenchmarkSpec,
    CommandPlan,
    MethodSpec,
    add_optional_path,
    add_optional_value,
    normalize_method,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

DRAKES_METHOD_ALIASES = {
    "pretrained": "pretrained",
    "Pretrained": "pretrained",
    "finetuned": "finetuned",
    "zero_alpha": "zero_alpha",
    "CFG": "CFG",
    "CG": "CG",
    "SMC": "SMC",
    "TDS": "TDS",
    "DOIT": "DOIT",
    "D-DOIT-Q": "DOIT",
    "DOIT_LBOK": "DOIT_LBOK",
    "D-DOIT-LBOK": "DOIT_LBOK",
    "D-DOIT-LBOK-S": "DOIT_LBOK",
}


def normalize_generation_method(method: str) -> str:
    return normalize_method(method, DRAKES_METHOD_ALIASES, "drakes")


def build_legacy_generate_command(args: object, extra_args: Sequence[str]) -> CommandPlan:
    """Build a DRAKES generation/evaluation command from unified CLI args."""

    method = normalize_generation_method(getattr(args, "method"))
    task = getattr(args, "task") or "hepg2"
    if task not in SPEC.supported_tasks:
        valid = ", ".join(SPEC.supported_tasks)
        raise ValueError(f"Unsupported DRAKES task '{task}'. Valid tasks: {valid}")

    num_sequences = getattr(args, "num_sequences")
    batch_size = getattr(args, "num_samples_per_batch") or min(num_sequences, 64)
    num_batches = max(1, math.ceil(num_sequences / batch_size))
    effective_sequences = num_batches * batch_size

    command = [
        getattr(args, "python"),
        "-m",
        "ddoit.benchmarks.drakes_generation",
        "--methods",
        method,
        "--num-sample-batches",
        str(num_batches),
        "--num-samples-per-batch",
        str(batch_size),
        "--output-dir",
        str(getattr(args, "output_dir")),
    ]
    add_optional_value(command, "--seed", getattr(args, "seed"))
    add_optional_path(command, "--ckpt-path", getattr(args, "ckpt_path"))
    add_optional_path(command, "--pretrained-path", getattr(args, "pretrained_path"))
    add_optional_path(command, "--zero-alpha-path", getattr(args, "zero_alpha_path"))
    add_optional_path(command, "--cfg-path", getattr(args, "cfg_path"))
    add_optional_value(command, "--doit-M", getattr(args, "doit_M"))
    add_optional_value(command, "--doit-omega", getattr(args, "doit_omega"))
    add_optional_value(command, "--doit-beta", getattr(args, "doit_beta"))
    add_optional_value(command, "--doit-alpha", getattr(args, "doit_alpha"))
    add_optional_value(command, "--doit-l-star", getattr(args, "doit_l_star"))
    add_optional_value(command, "--doit-gamma", getattr(args, "doit_gamma"))
    command.extend(extra_args)

    notes: list[str] = []
    if effective_sequences != num_sequences:
        notes.append(
            "DRAKES eval samples whole batches; "
            f"requested {num_sequences}, command will sample {effective_sequences}."
        )

    return CommandPlan(command=tuple(command), cwd=REPO_ROOT, notes=tuple(notes))


SPEC = BenchmarkSpec(
    name="drakes",
    display_name="DRAKES/HepG2 enhancer design",
    task_family="single-target enhancer design",
    adapter_module="ddoit.benchmarks.drakes",
    legacy_paths=(),
    supported_tasks=("hepg2",),
    methods=(
        MethodSpec("Pretrained", "ddoit.benchmarks.drakes_diffusion_gosai_update"),
        MethodSpec("CG", "ddoit.benchmarks.drakes_diffusion_cg"),
        MethodSpec("SMC", "ddoit.benchmarks.drakes_diffusion_smc"),
        MethodSpec("TDS", "ddoit.benchmarks.drakes_diffusion_tds"),
        MethodSpec("D-DOIT-Q", "ddoit.benchmarks.drakes_diffusion_doit"),
        MethodSpec("D-DOIT-LBOK-S", "ddoit.benchmarks.drakes_doit_late_bok"),
    ),
    stable_result_paths=(),
    paper_locations=(),
    notes=(
        "DRAKES generation and evaluation run through the package-owned ddoit "
        "benchmark implementation."
    ),
)
