"""DRAKES model runtime adapters.

This module centralizes DRAKES runtime imports so lightweight ``ddoit`` metadata
can be imported without loading GPU-heavy dependencies at module import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable

@dataclass(frozen=True)
class DrakesModelModules:
    diffusion_gosai_cfg: Any
    diffusion_gosai_update: Any
    diffusion_doit: Any
    doit_late_bok: Any
    diffusion_cg: Any
    diffusion_smc: Any
    diffusion_tds: Any
    set_seed: Callable[..., None]


def load_model_modules(legacy_root: str | Path | None = None) -> DrakesModelModules:
    """Import package-owned DRAKES model modules lazily.

    ``legacy_root`` is accepted for compatibility with earlier internal
    migration calls and is ignored.
    """

    from ddoit.benchmarks import drakes_noise
    from ddoit.benchmarks import drakes_models_pkg
    from ddoit.benchmarks import drakes_base_config
    from ddoit.benchmarks import drakes_dataloader
    from ddoit.benchmarks import drakes_diffusion_cg
    from ddoit.benchmarks import drakes_diffusion_doit
    from ddoit.benchmarks import drakes_diffusion_gosai_cfg
    from ddoit.benchmarks import drakes_diffusion_gosai_update
    from ddoit.benchmarks import drakes_diffusion_smc
    from ddoit.benchmarks import drakes_diffusion_tds
    from ddoit.benchmarks import drakes_doit_late_bok
    from ddoit.benchmarks import drakes_oracle_runtime
    from ddoit.benchmarks import drakes_utils

    sys.modules["base_config"] = drakes_base_config
    sys.modules["dataloader_gosai"] = drakes_dataloader
    sys.modules["diffusion_cg"] = drakes_diffusion_cg
    sys.modules["diffusion_doit"] = drakes_diffusion_doit
    sys.modules["diffusion_gosai_cfg"] = drakes_diffusion_gosai_cfg
    sys.modules["diffusion_gosai_update"] = drakes_diffusion_gosai_update
    sys.modules["diffusion_smc"] = drakes_diffusion_smc
    sys.modules["diffusion_tds"] = drakes_diffusion_tds
    sys.modules["doit_late_bok"] = drakes_doit_late_bok
    sys.modules["noise_schedule"] = drakes_noise
    sys.modules["oracle"] = drakes_oracle_runtime
    sys.modules["utils"] = drakes_utils
    sys.modules["models"] = drakes_models_pkg
    sys.modules["models.dnaconv"] = drakes_models_pkg.dnaconv
    sys.modules["models.ema"] = drakes_models_pkg.ema

    return DrakesModelModules(
        diffusion_gosai_cfg=drakes_diffusion_gosai_cfg,
        diffusion_gosai_update=drakes_diffusion_gosai_update,
        diffusion_doit=drakes_diffusion_doit,
        doit_late_bok=drakes_doit_late_bok,
        diffusion_cg=drakes_diffusion_cg,
        diffusion_smc=drakes_diffusion_smc,
        diffusion_tds=drakes_diffusion_tds,
        set_seed=drakes_utils.set_seed,
    )
