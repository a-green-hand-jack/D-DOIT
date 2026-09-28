"""DRAKES path helpers for data, checkpoints, and package assets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

from ddoit.benchmarks.drakes import REPO_ROOT


DEFAULT_BASE_PATH = REPO_ROOT / "data_and_model"
CONFIG_DIR = Path(__file__).resolve().parent / "drakes_configs_gosai"


def base_path() -> Path:
    """Return the DRAKES data/model root, honoring ``DRAKES_BASE_PATH``."""

    return Path(os.getenv("DRAKES_BASE_PATH", str(DEFAULT_BASE_PATH))).expanduser()


def get_path(relative_path: Union[str, Path]) -> Path:
    """Resolve a path relative to the DRAKES data/model root."""

    return base_path() / relative_path


def config_dir() -> Path:
    """Return the package-owned DRAKES Hydra config directory."""

    return CONFIG_DIR
