"""Shared D-DOIT core interfaces and, eventually, guidance implementations."""

from ddoit.core.commands import (
    CommandPlan,
    add_optional_path,
    add_optional_value,
    normalize_method,
)
from ddoit.core.protocols import (
    BenchmarkAdapter,
    BenchmarkSpec,
    GenerationRequest,
    GenerationResult,
    MethodSpec,
)

__all__ = [
    "BenchmarkAdapter",
    "BenchmarkSpec",
    "CommandPlan",
    "GenerationRequest",
    "GenerationResult",
    "MethodSpec",
    "add_optional_path",
    "add_optional_value",
    "normalize_method",
]
