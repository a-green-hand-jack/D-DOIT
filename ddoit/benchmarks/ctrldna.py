"""Ctrl-DNA benchmark metadata and generation command planning.

The package-owned runtime entrypoint lives at
``ddoit.benchmarks.ctrldna_generation``. The heavy MDLM model implementation is
still imported from the retained legacy Ctrl-DNA repository while migration
continues.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
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


CODE_REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_REPO_FALLBACK = Path("external/Ctrl-DNA")


def resolve_legacy_repo(code_repo_root: Path = CODE_REPO_ROOT) -> Path:
    """Resolve the legacy Ctrl-DNA repo without assuming private paths."""

    override = os.environ.get("DDOIT_LEGACY_CTRLDNA_ROOT")
    if override:
        return Path(override).expanduser()

    candidates = (
        code_repo_root.parent / "CtrlDNA/Ctrl-DNA",
        code_repo_root.parents[1] / "CtrlDNA/Ctrl-DNA",
        code_repo_root / LEGACY_REPO_FALLBACK,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return code_repo_root / LEGACY_REPO_FALLBACK


LEGACY_REPO = resolve_legacy_repo()


def ensure_legacy_repo_on_path(legacy_root: str | Path = LEGACY_REPO) -> Path:
    """Make legacy Ctrl-DNA modules importable for transitional adapters."""

    root = Path(legacy_root).expanduser().resolve()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def resolve_project_root(legacy_root: str | Path = LEGACY_REPO) -> Path:
    """Return the CtrlDNA project root that contains data, checkpoints, and logs."""

    return Path(legacy_root).expanduser().resolve().parent


def default_data_root(project_root: str | Path | None = None) -> Path:
    root = Path(project_root) if project_root is not None else resolve_project_root()
    return root / "data"


def default_checkpoint_root(project_root: str | Path | None = None) -> Path:
    root = Path(project_root) if project_root is not None else resolve_project_root()
    return root / "checkpoints"


def default_tfbs_dir(project_root: str | Path | None = None) -> Path:
    return default_data_root(project_root) / "motifs"

CTRLDNA_METHOD_ALIASES = {
    "pretrained": "unconditional",
    "Pretrained": "unconditional",
    "unconditional": "unconditional",
    "CG": "CG",
    "SMC": "SMC",
    "TDS": "TDS",
    "DOIT_LBOK": "DOIT_LBOK",
    "D-DOIT-LBOK": "DOIT_LBOK",
    "D-DOIT-LBOK-S": "DOIT_LBOK",
}


@dataclass(frozen=True)
class CtrlDNATaskSpec:
    """Dependency-free Ctrl-DNA task metadata used by the unified adapter."""

    name: str
    seq_len: int
    min_fitness: float
    max_fitness: float
    prefix_label: str
    panel: tuple[str, str, str]
    start_relpath: str


TASK_SPECS: dict[str, CtrlDNATaskSpec] = {
    "hepg2": CtrlDNATaskSpec(
        name="hepg2",
        seq_len=200,
        min_fitness=-6.051336,
        max_fitness=10.992575,
        prefix_label="100",
        panel=("hepg2", "k562", "sknsh"),
        start_relpath="human/{task}_{level}.csv",
    ),
    "k562": CtrlDNATaskSpec(
        name="k562",
        seq_len=200,
        min_fitness=-5.857445,
        max_fitness=10.781755,
        prefix_label="010",
        panel=("k562", "hepg2", "sknsh"),
        start_relpath="human/{task}_{level}.csv",
    ),
    "sknsh": CtrlDNATaskSpec(
        name="sknsh",
        seq_len=200,
        min_fitness=-7.283977,
        max_fitness=12.888308,
        prefix_label="001",
        panel=("sknsh", "hepg2", "k562"),
        start_relpath="human/{task}_{level}.csv",
    ),
}


def get_task_spec(task: str) -> CtrlDNATaskSpec:
    try:
        return TASK_SPECS[task]
    except KeyError as exc:
        valid = ", ".join(sorted(TASK_SPECS))
        raise ValueError(f"Unsupported Ctrl-DNA task '{task}'. Valid tasks: {valid}") from exc


def get_task_cells(task: str) -> tuple[str, str, str]:
    return get_task_spec(task).panel


def get_fitness_info(task: str) -> tuple[int, float, float]:
    spec = get_task_spec(task)
    return spec.seq_len, spec.min_fitness, spec.max_fitness


def normalize_score(score: float, task: str) -> float:
    _, min_fitness, max_fitness = get_fitness_info(task)
    return (score - min_fitness) / (max_fitness - min_fitness)


def default_starting_sequences_path(
    task: str,
    *,
    level: str = "hard",
    data_root: str | Path,
) -> Path:
    spec = get_task_spec(task)
    return Path(data_root) / spec.start_relpath.format(task=task, level=level)


def default_oracle_checkpoint_path(
    task: str,
    *,
    checkpoint_root: str | Path,
    oracle_type: str = "paired",
) -> Path:
    if oracle_type not in {"paired", "separate", "dlow"}:
        raise ValueError(f"Unsupported oracle type: {oracle_type}")
    get_task_spec(task)
    subdir = {
        "paired": "oracle_train",
        "separate": "oracle_eval",
        "dlow": "oracle_eval",
    }[oracle_type]
    return Path(checkpoint_root) / subdir / f"human_regression_{oracle_type}_{task}.ckpt"


def normalize_generation_method(method: str) -> str:
    return normalize_method(method, CTRLDNA_METHOD_ALIASES, "ctrldna")


def _require_generate_paths(args: object) -> None:
    missing = []
    for field in ("checkpoint", "checkpoint_root", "output", "summary_output"):
        if getattr(args, field) is None:
            missing.append("--" + field.replace("_", "-"))
    if missing:
        raise ValueError("Ctrl-DNA generation requires " + ", ".join(missing) + ".")


def build_legacy_generate_command(args: object, extra_args: Sequence[str]) -> CommandPlan:
    """Build a package-owned Ctrl-DNA generation command from unified CLI args."""

    method = normalize_generation_method(getattr(args, "method"))
    task = getattr(args, "task") or "hepg2"
    get_task_spec(task)
    _require_generate_paths(args)

    legacy_root = getattr(args, "legacy_ctrldna_root")
    command = [
        getattr(args, "python"),
        "-m",
        "ddoit.benchmarks.ctrldna_generation",
        "--legacy-ctrldna-root",
        str(legacy_root),
        "--checkpoint",
        str(getattr(args, "checkpoint")),
        "--checkpoint-root",
        str(getattr(args, "checkpoint_root")),
        "--task",
        task,
        "--method",
        method,
        "--num-sequences",
        str(getattr(args, "num_sequences")),
        "--output",
        str(getattr(args, "output")),
        "--summary-output",
        str(getattr(args, "summary_output")),
    ]
    add_optional_value(command, "--seed", getattr(args, "seed"))
    add_optional_value(command, "--num-steps", getattr(args, "num_steps"))
    add_optional_value(command, "--beta", getattr(args, "beta"))
    add_optional_value(command, "--resample-every", getattr(args, "resample_every"))
    add_optional_value(command, "--guidance-scale", getattr(args, "guidance_scale"))
    add_optional_value(command, "--K", getattr(args, "K"))
    add_optional_value(command, "--bok-split-step", getattr(args, "bok_split_step"))
    add_optional_value(command, "--doit-M", getattr(args, "doit_M"))
    add_optional_value(command, "--doit-omega", getattr(args, "doit_omega"))
    add_optional_value(command, "--doit-beta", getattr(args, "doit_beta"))
    add_optional_value(command, "--doit-alpha", getattr(args, "doit_alpha"))
    add_optional_value(command, "--doit-l-star", getattr(args, "doit_l_star"))
    add_optional_value(command, "--doit-gamma", getattr(args, "doit_gamma"))
    add_optional_value(command, "--batch-size", getattr(args, "batch_size"))
    add_optional_value(command, "--reward-mode", getattr(args, "reward_mode"))
    add_optional_value(command, "--reward-lambda", getattr(args, "reward_lambda"))
    add_optional_value(command, "--device", getattr(args, "device"))
    add_optional_path(command, "--reference-csv", getattr(args, "reference_csv"))
    add_optional_value(command, "--reference-col", getattr(args, "reference_col"))
    add_optional_value(command, "--reference-n", getattr(args, "reference_n"))
    command.extend(extra_args)

    return CommandPlan(command=tuple(command), cwd=CODE_REPO_ROOT)


SPEC = BenchmarkSpec(
    name="ctrldna",
    display_name="Ctrl-DNA cell-type-specific enhancer design",
    task_family="cell-type-specific enhancer design",
    adapter_module="ddoit.benchmarks.ctrldna",
    legacy_paths=(LEGACY_REPO,),
    supported_tasks=("hepg2", "k562", "sknsh"),
    methods=(
        MethodSpec("Pretrained", "ctrl_dna.mdlm.diffusion"),
        MethodSpec("CG", "ctrl_dna.mdlm.guided_diffusion"),
        MethodSpec("SMC", "ctrl_dna.mdlm.guided_diffusion"),
        MethodSpec("TDS", "ctrl_dna.mdlm.guided_diffusion"),
        MethodSpec("D-DOIT-Q", "ctrl_dna.mdlm.guided_diffusion"),
        MethodSpec("D-DOIT-LBOK-S", "ctrl_dna.mdlm.guided_diffusion"),
    ),
    stable_result_paths=(),
    paper_locations=(),
    notes=(
        "Ctrl-DNA generation, reward, and evaluation now enter through ddoit. "
        "The retained legacy repository still supplies heavy MDLM model code "
        "until that runtime is migrated."
    ),
)
