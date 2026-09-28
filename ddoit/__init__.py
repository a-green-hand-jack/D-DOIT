"""Unified D-DOIT code namespace.

This package is the migration target for shared D-DOIT guidance code and
benchmark adapters. DRAKES runs natively under ``ddoit``; Ctrl-DNA can still
use an explicitly provided external legacy checkout while migration continues.
"""

from ddoit.benchmarks.registry import BENCHMARKS, get_benchmark

__all__ = ["BENCHMARKS", "get_benchmark"]
