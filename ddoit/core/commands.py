"""Lightweight command planning helpers for legacy compatibility adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommandPlan:
    """A runnable command plus the directory where it should run."""

    command: tuple[str, ...]
    cwd: Path
    notes: tuple[str, ...] = ()


def add_optional_path(command: list[str], flag: str, value: Path | None) -> None:
    if value is not None:
        command.extend((flag, str(value)))


def add_optional_value(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend((flag, str(value)))


def normalize_method(method: str, aliases: dict[str, str], benchmark: str) -> str:
    try:
        return aliases[method]
    except KeyError as exc:
        valid = ", ".join(sorted(aliases))
        raise ValueError(
            f"Unsupported method '{method}' for benchmark '{benchmark}'. "
            f"Valid methods/aliases: {valid}"
        ) from exc
