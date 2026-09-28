"""
DiffusionCG – Classifier Guidance (CG) controlled sampling for MDLM.

At each denoising step a gradient of the reward function with respect to the
one-hot input is computed and used to bias the transition distribution:

    q̃(x_{t-1} | x_t)  ∝  q(x_{t-1} | x_t) · exp( γ · ∂r / ∂x_onehot )

The gradient is evaluated at the *expected* clean state E[x₀ | x_t], so no
differentiable denoiser is required — only the reward model must be
differentiable w.r.t. its one-hot float input.

References
----------
- diffusion_gosai_update.py  (original implementation in the Diffusion base class)
"""

import torch
import torch.nn.functional as F

from ddoit.benchmarks.drakes_diffusion_gosai_update import (
    Diffusion,
    _sample_categorical,
)


class DiffusionCG(Diffusion):
    """Diffusion model with Classifier Guidance (CG).

    Inherits the full MDLM diffusion model and overrides only the
    controlled-sampling methods.  Call via ``controlled_sample_CG(...)`` —
    compatible with ``eval.py``'s ``sample_controlled(model, "CG", ...)``.
    """

    # ------------------------------------------------------------------
    # Gradient computation
    # ------------------------------------------------------------------

    def compute_gradient_CG(self, x_onehot, x, reward_model, sigma_s):
        """Compute ∂r(E[x₀|x_t]) / ∂x_onehot.

        Args:
            x_onehot:     FloatTensor [B, L, V] — one-hot (requires_grad=True).
            x:            LongTensor  [B, L]    — current discrete state x_t.
            reward_model: differentiable reward oracle.
            sigma_s:      noise level at the *next* timestep s = t − dt.

        Returns:
            x_grad: FloatTensor [B, L, V] — gradient tensor.
        """
        x_onehot.requires_grad_(True)
        expected_x0 = self.forward(x_onehot, sigma_s)
        scores = reward_model(expected_x0.transpose(1, 2)[:, 0:4, :])[:, 0]
        scores = scores.mean()
        scores.backward()
        x_grad = x_onehot.grad.clone()
        return x_grad

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def _ddpm_update_controlled_CG(self, x, t, dt, reward_model, guidance_scale):
        """Single denoising step with Classifier Guidance.

        Args:
            x:              LongTensor [B, L] — current discrete state x_t.
            t:              Tensor [B, 1] — current timestep.
            dt:             float — timestep delta.
            reward_model:   differentiable reward oracle.
            guidance_scale: float — multiplier for the gradient signal.

        Returns:
            x_next: LongTensor [B, L] — next discrete state x_{t-1}.
        """
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape

        move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)
        move_chance_t = move_chance_t[:, None, None]
        move_chance_s = move_chance_s[:, None, None]

        unet_conditioning = sigma_t
        log_p_x0 = self.forward(x, unet_conditioning)
        assert move_chance_t.ndim == log_p_x0.ndim

        q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)
        x_onehot = F.one_hot(x, num_classes=5).float()

        x_grad = self.compute_gradient_CG(x_onehot, x, reward_model, sigma_s)
        guidance = guidance_scale * (
            x_grad - x_grad[:, :, self.mask_index][:, :, None]
        )
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        q_xs = q_xs * guidance.exp()

        _x = _sample_categorical(q_xs)
        copy_flag = (x != self.mask_index).to(x.dtype)
        return copy_flag * x + (1 - copy_flag) * _x

    # ------------------------------------------------------------------
    # Sampling loop
    # ------------------------------------------------------------------

    def controlled_sample_CG(
        self,
        reward_model,
        guidance_scale,
        *,
        num_steps=None,
        eps=1e-5,
        eval_sp_size=None,
    ):
        """Generate samples using Classifier Guidance.

        Compatible with ``eval.py``'s
        ``sample_controlled(model, "CG", reward_model, ...)``.

        Args:
            reward_model:   differentiable reward oracle.
            guidance_scale: float — gradient multiplier.
            num_steps:      total denoising steps (default from config).
            eps:            small constant to avoid t=0 (default 1e-5).
            eval_sp_size:   batch size override (default from config).

        Returns:
            x: LongTensor [B, L] — generated discrete sequences.
        """
        if eval_sp_size is None:
            batch_size_per_gpu = self.config.loader.eval_batch_size
        else:
            batch_size_per_gpu = eval_sp_size

        if self.parameterization == 'ar':
            return self._ar_sampler(batch_size_per_gpu)

        if num_steps is None:
            num_steps = self.config.sampling.steps

        x = self._sample_prior(
            batch_size_per_gpu,
            self.config.model.length,
        ).to(self.device)

        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps

        for i in range(num_steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == 'ddpm':
                x = self._ddpm_update_controlled_CG(x, t, dt, reward_model, guidance_scale)
            else:
                x = self._analytic_update(x, t, dt)

        if self.config.sampling.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == 'analytic':
                x = self._denoiser_update(x, t)
            else:
                unet_conditioning = self.noise(t)[0]
                logits = self.forward(x, unet_conditioning)
                x = logits[:, :, :-1].argmax(dim=-1)

        return x
