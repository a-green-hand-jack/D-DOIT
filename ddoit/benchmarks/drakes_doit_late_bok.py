"""
DiffusionDOITLateBok – DOIT with late-stage best-of-K.

Run B sequences normally for the first N steps, then duplicate each
sequence into K copies for the remaining steps.  At the end, each
sequence picks its best copy.  This avoids the 2x cost of full bok
while capturing the quality benefit where it matters most (late steps).

Usage:
    model = DiffusionDOITLateBok.load_from_checkpoint(path, config=cfg)
    samples = model.controlled_sample_DOIT(
        reward_model, M=8, omega=4.0, bok_split_step=110, best_of_k=2, ...)
"""

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from ddoit.benchmarks.drakes_diffusion_doit import DiffusionDOIT
from ddoit.benchmarks.drakes_diffusion_gosai_update import (
    Diffusion,
    _sample_categorical,
)


class DiffusionDOITLateBok(DiffusionDOIT):
    """DOIT with late-stage trajectory splitting for best-of-K."""

    def controlled_sample_DOIT(
        self,
        reward_model,
        *,
        l_star: int = 128,
        l_end: int = 1,
        M: int = 8,
        omega: float = 4.0,
        beta: float = 1.0,
        alpha: float = 3.0,
        gamma: float = 1.2,
        best_of_k: int = 2,
        bok_split_step: int = 110,
        verbose: bool = False,
        num_steps: int = None,
        eps: float = 1e-5,
        eval_sp_size: int = None,
        # Pass through to _ddpm_update_controlled_DOIT
        M_schedule: str = "fixed",
        M_min: int = 4,
        M_thresh_high: int = 80,
        M_thresh_low: int = 20,
        omega_schedule: str = "fixed",
        omega_min: float = 0.5,
        omega_thresh_high: int = 80,
        omega_thresh_low: int = 20,
        phase_strategy: str = "none",
        phase_skip_thresh: int = 100,
        phase_active_thresh: int = 20,
        phase_M_refine: int = 8,
        phase_omega_refine_scale: float = 1.5,
        # Ignored compatibility params
        diagnostics_sink: list = None,
        candidate_sink: list = None,
        completion_noise_K: int = 0,
        completion_noise_every: int = 5,
        resample_strategy: str = "per_sequence",
        **kwargs,
    ) -> torch.Tensor:
        """Generate sequences with late-stage best-of-K.

        Args:
            bok_split_step: step index at which to duplicate sequences.
                0 = full bok from start (equivalent to original bok).
                110 = split at step 110, only last 18 steps with K copies.
                128 = never split (equivalent to no bok).
            best_of_k: number of copies per sequence after split.
        """
        batch_size = eval_sp_size or self.config.loader.eval_batch_size
        K = max(1, best_of_k)
        num_steps = num_steps or self.config.sampling.steps
        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps

        # Clamp split step
        split = min(max(0, bok_split_step), num_steps)

        x = self._sample_prior(
            batch_size, self.config.model.length).to(self.device)

        # Common kwargs for _ddpm_update_controlled_DOIT
        step_kwargs = dict(
            M_schedule=M_schedule, M_min=M_min,
            M_thresh_high=M_thresh_high, M_thresh_low=M_thresh_low,
            omega_schedule=omega_schedule, omega_min=omega_min,
            omega_thresh_high=omega_thresh_high, omega_thresh_low=omega_thresh_low,
            phase_strategy=phase_strategy,
            phase_skip_thresh=phase_skip_thresh,
            phase_active_thresh=phase_active_thresh,
            phase_M_refine=phase_M_refine,
            phase_omega_refine_scale=phase_omega_refine_scale,
        )

        for i in range(num_steps):
            # ── Split point: duplicate sequences ────────────────────────
            if i == split and K > 1:
                # [B, L] → [B*K, L]
                x = x.unsqueeze(1).expand(
                    batch_size, K, -1).contiguous().view(batch_size * K, -1)
                if verbose:
                    print(f"[LateBok] Split at step {i}: "
                          f"B={batch_size} → B*K={batch_size * K}")

            current_bs = x.shape[0]
            t = timesteps[i] * torch.ones(current_bs, 1, device=self.device)

            if self.sampler == 'ddpm':
                x = self._ddpm_update_controlled_DOIT(
                    x, t, dt, reward_model,
                    step_idx=i, num_steps=num_steps,
                    l_star=l_star, l_end=l_end,
                    M=M, omega=omega, beta=beta,
                    alpha=alpha, gamma=gamma,
                    verbose=False,
                    **step_kwargs,
                )
            else:
                x = self._analytic_update(x, t, dt)

        # Final denoising
        if self.config.sampling.noise_removal:
            current_bs = x.shape[0]
            t = timesteps[-1] * torch.ones(current_bs, 1, device=self.device)
            if self.sampler == 'analytic':
                x = self._denoiser_update(x, t)
            else:
                unet_cond = self.noise(t)[0]
                logits = self.forward(x, unet_cond)
                x = logits[:, :, :-1].argmax(dim=-1)

        # ── Best-of-K selection ─────────────────────────────────────────
        if K > 1 and x.shape[0] == batch_size * K:
            acgt = self.mask_index
            oh = F.one_hot(x, num_classes=acgt).float()
            scores = reward_model(oh.transpose(1, 2))
            if scores.ndim > 1:
                scores = scores[:, 0]
            scores = scores.flatten()
            x = x.view(batch_size, K, -1)
            scores = scores.view(batch_size, K)
            best_k = scores.argmax(dim=1)
            x = x[torch.arange(batch_size, device=x.device), best_k]

        return x
