"""Shared interfaces for D-DOIT benchmark adapters.

The first migration step is to make benchmark ownership explicit. These
dataclasses are intentionally lightweight and dependency-free so they can be
imported in local tooling without requiring the heavy GPU environments used by
the benchmark implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol, Sequence


@dataclass(frozen=True)
class MethodSpec:
    """Paper-facing method name and benchmark-specific default parameters."""

    name: str
    implementation: str
    default_params: Mapping[str, object] = field(default_factory=dict)
    notes: str = ""


@dataclass(frozen=True)
class BenchmarkSpec:
    """Static metadata for one benchmark adapter."""

    name: str
    display_name: str
    task_family: str
    adapter_module: str
    legacy_paths: tuple[Path, ...]
    supported_tasks: tuple[str, ...]
    methods: tuple[MethodSpec, ...]
    stable_result_paths: tuple[Path, ...]
    paper_locations: tuple[Path, ...]
    notes: str = ""


@dataclass(frozen=True)
class GenerationRequest:
    """Minimal benchmark-independent generation request."""

    benchmark: str
    method: str
    num_sequences: int
    task: str | None = None
    seed: int | None = None
    checkpoint: Path | None = None
    output: Path | None = None
    params: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResult:
    """Minimal benchmark-independent generation result pointer."""

    benchmark: str
    method: str
    task: str | None
    sequences_path: Path | None
    summary_path: Path | None
    metrics: Mapping[str, object] = field(default_factory=dict)
    provenance: Mapping[str, object] = field(default_factory=dict)


class BenchmarkAdapter(Protocol):
    """Protocol for future executable benchmark adapters."""

    spec: BenchmarkSpec

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate sequences for the benchmark."""

    def evaluate(self, sequences: Sequence[str], request: GenerationRequest) -> GenerationResult:
        """Evaluate generated sequences for the benchmark."""
