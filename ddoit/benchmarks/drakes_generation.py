"""DRAKES generation runner.

This module gives the unified ``ddoit`` CLI a package-owned runtime entrypoint
for DRAKES generation and evaluation.
"""

from __future__ import annotations

from typing import Sequence

def main(argv: Sequence[str] | None = None) -> None:
    """Run DRAKES generation/evaluation behind the ddoit module boundary."""

    from ddoit.benchmarks.drakes_evaluator import main as evaluate_main

    evaluate_main(argv)


if __name__ == "__main__":
    main()
