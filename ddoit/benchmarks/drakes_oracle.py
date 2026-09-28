"""DRAKES oracle adapters."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _oracle_runtime(legacy_root: str | Path | None = None) -> Any:
    """Load the ddoit-owned DRAKES oracle runtime.

    ``legacy_root`` is accepted for source compatibility with earlier migration
    steps and is ignored.
    """

    from ddoit.benchmarks import drakes_oracle_runtime

    return drakes_oracle_runtime


def load_reward_model(mode: str = "train", *, legacy_root: str | Path | None = None) -> Any:
    """Load the DRAKES GOSAI oracle used as reward or eval model."""

    return _oracle_runtime(legacy_root).get_gosai_oracle(mode=mode)


def predict_activity(
    seqs: Sequence[str],
    *,
    model: Any = None,
    mode: str = "eval",
    legacy_root: str | Path | None = None,
) -> Any:
    """Predict HepG2/K562/SKN-SH activity with the DRAKES GOSAI oracle."""

    return _oracle_runtime(legacy_root).cal_gosai_pred_new(
        seqs,
        model=model,
        mode=mode,
    )


def predict_atac(
    seqs: Sequence[str],
    *,
    model: Any = None,
    legacy_root: str | Path | None = None,
) -> Any:
    """Predict ATAC accessibility with the DRAKES ATAC oracle."""

    return _oracle_runtime(legacy_root).cal_atac_pred_new(seqs, model=model)


def high_expression_kmers(
    *,
    k: int = 3,
    return_clss: bool = False,
    legacy_root: str | Path | None = None,
) -> Any:
    """Return high-expression reference k-mers from the DRAKES data."""

    return _oracle_runtime(legacy_root).cal_highexp_kmers(
        k=k,
        return_clss=return_clss,
    )
