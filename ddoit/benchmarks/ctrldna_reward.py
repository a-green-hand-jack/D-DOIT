"""Ctrl-DNA tensor reward adapters for guided diffusion.

This module intentionally keeps heavy imports optional. The benchmark registry
does not import it, so lightweight tooling can still run without torch or the
legacy Ctrl-DNA dependencies installed.
"""

from __future__ import annotations

from pathlib import Path

try:  # pragma: no cover - exercised only when the full Ctrl-DNA stack is present.
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - local dev may not have torch.
    torch = None
    nn = None

from ddoit.benchmarks.ctrldna import (
    default_oracle_checkpoint_path,
    ensure_legacy_repo_on_path,
    get_fitness_info,
    get_task_cells,
    resolve_legacy_repo,
)


def _require_torch():
    if torch is None or nn is None:
        raise RuntimeError(
            "Ctrl-DNA reward adapters require torch. Activate the Ctrl-DNA "
            "environment before importing or constructing rewards."
        )
    return torch, nn


def _load_enformer_model(legacy_root: str | Path | None = None):
    ensure_legacy_repo_on_path(legacy_root or resolve_legacy_repo())
    from ctrl_dna.src.reglm.regression import EnformerModel

    return EnformerModel


_BaseModule = nn.Module if nn is not None else object


class EnformerTensorReward(_BaseModule):
    """Single-cell Ctrl-DNA reward adapter with tensor input and tensor output."""

    MAX_REWARD_BATCH = 256

    def __init__(
        self,
        model,
        min_fitness: float,
        max_fitness: float,
    ) -> None:
        _require_torch()
        super().__init__()
        self.model = model
        self.min_fitness = min_fitness
        self.max_fitness = max_fitness

    def forward(self, x):
        """Return normalized activity scores for soft one-hot `[B, 4, L]` input."""

        torch_mod, _ = _require_torch()
        if x.shape[0] > self.MAX_REWARD_BATCH:
            return torch_mod.cat(
                [
                    self.forward(x[i : i + self.MAX_REWARD_BATCH])
                    for i in range(0, x.shape[0], self.MAX_REWARD_BATCH)
                ],
                dim=0,
            )
        x = x.transpose(1, 2).to(next(self.model.parameters()).device)
        h = self.model.trunk(x)
        h = self.model.head(h)
        h = h.mean(dim=1)
        if self.model.loss_type == "poisson":
            h = torch_mod.exp(h)
        score = h.squeeze(-1)
        return (score - self.min_fitness) / (self.max_fitness - self.min_fitness)


class PairedOracleTensorReward(_BaseModule):
    """Weighted three-cell Ctrl-DNA reward used by legacy guided generation."""

    def __init__(
        self,
        adapters: dict[str, EnformerTensorReward],
        task: str,
        weights: tuple[float, float, float] | None = None,
    ) -> None:
        _, nn_mod = _require_torch()
        super().__init__()
        self.adapters = nn_mod.ModuleDict(adapters)
        self.task = task
        self.panel = get_task_cells(task)
        self.weights = weights if weights is not None else (0.8, -0.1, -0.1)

    def forward(self, x):
        scores = [self.adapters[cell](x) for cell in self.panel]
        return (
            self.weights[0] * scores[0]
            + self.weights[1] * scores[1]
            + self.weights[2] * scores[2]
        )


class CompositeMaxReward(_BaseModule):
    """Paper-spec reward: `R_c(x) - lambda * max_{c' != c} R_{c'}(x)`."""

    def __init__(
        self,
        adapters: dict[str, EnformerTensorReward],
        task: str,
        lambda_: float,
    ) -> None:
        _, nn_mod = _require_torch()
        super().__init__()
        self.adapters = nn_mod.ModuleDict(adapters)
        self.task = task
        self.panel = get_task_cells(task)
        self.lambda_ = float(lambda_)

    def forward(self, x):
        torch_mod, _ = _require_torch()
        target = self.adapters[self.panel[0]](x)
        offs = torch_mod.stack([self.adapters[cell](x) for cell in self.panel[1:]], dim=-1)
        worst_off, _ = offs.max(dim=-1)
        return target - self.lambda_ * worst_off


def load_paired_reward(
    task: str,
    checkpoint_root: str | Path,
    device: str | object = "cuda",
    weights: tuple[float, float, float] | None = None,
    *,
    reward_mode: str = "weighted",
    lambda_: float = 1.0,
    legacy_root: str | Path | None = None,
):
    """Load Ctrl-DNA paired oracle checkpoints as a tensor reward module."""

    torch_mod, _ = _require_torch()
    enformer_model = _load_enformer_model(legacy_root)
    panel = get_task_cells(task)
    adapters: dict[str, EnformerTensorReward] = {}

    for cell in panel:
        ckpt_path = default_oracle_checkpoint_path(
            task=cell,
            checkpoint_root=checkpoint_root,
            oracle_type="paired",
        )
        model = enformer_model.load_from_checkpoint(
            str(ckpt_path),
            map_location=device,
        ).to(device)
        model.eval()

        _, min_fit, max_fit = get_fitness_info(cell)
        adapters[cell] = EnformerTensorReward(model, min_fit, max_fit)

    if reward_mode == "composite":
        return CompositeMaxReward(adapters, task=task, lambda_=lambda_)
    if reward_mode != "weighted":
        raise ValueError(f"unknown reward_mode={reward_mode!r}")
    return PairedOracleTensorReward(adapters, task=task, weights=weights)
