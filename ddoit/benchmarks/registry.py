"""Registry for D-DOIT benchmark adapters."""

from __future__ import annotations

from ddoit.benchmarks.ctrldna import SPEC as CTRLDNA_SPEC
from ddoit.benchmarks.drakes import SPEC as DRAKES_SPEC
from ddoit.core import BenchmarkSpec


BENCHMARKS: dict[str, BenchmarkSpec] = {
    DRAKES_SPEC.name: DRAKES_SPEC,
    CTRLDNA_SPEC.name: CTRLDNA_SPEC,
}


def get_benchmark(name: str) -> BenchmarkSpec:
    """Return benchmark metadata by name."""
    try:
        return BENCHMARKS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(BENCHMARKS))
        raise ValueError(f"Unknown benchmark '{name}'. Valid benchmarks: {valid}") from exc
