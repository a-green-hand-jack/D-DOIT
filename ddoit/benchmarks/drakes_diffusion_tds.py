"""
DiffusionTDS – Twisted Diffusion Sampler (TDS) controlled sampling for MDLM.

TDS combines Classifier Guidance (CG) with SMC resampling.  The CG gradient
biases the proposal distribution, and the resulting importance weights correct
for the resulting distribution mismatch:

    w ∝ exp( (v_{t-1}(x_{t-1}) − v_t(x_t)) / α ) / ∏_i g(x_{t-1,i} | x_{t,i})

where g(·) is the guidance multiplier applied to each unmasked token.

References
----------
- diffusion_gosai_update.py  (original implementation in the Diffusion base class)
- diffusion_cg.py            (CG gradient helper)
"""

import numpy as np
import torch
import torch.nn.functional as F

from ddoit.benchmarks.drakes_diffusion_gosai_update import (
    Diffusion,
    _sample_categorical,
)


class DiffusionTDS(Diffusion):
    """Diffusion model with Twisted Diffusion Sampler (TDS) guidance.

    Inherits the full MDLM diffusion model and overrides only the
    controlled-sampling methods.  Call via ``controlled_sample_TDS(...)`` —
    compatible with ``eval.py``'s ``sample_controlled(model, "TDS", ...)``.
    """

    # ------------------------------------------------------------------
    # Gradient computation (same as CG)
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

    def _ddpm_update_controlled_TDS(
        self, x, t, dt, reward_model, alpha=1.0, guidance_scale=1000
    ):
        """Single denoising step with TDS guidance (CG proposal + SMC weights).

        Args:
            x:              LongTensor [B, L] — current discrete state x_t.
            t:              Tensor [B, 1] — current timestep.
            dt:             float — timestep delta.
            reward_model:   differentiable reward oracle.
            alpha:          float — SMC temperature.
            guidance_scale: float — CG gradient multiplier for the proposal.

        Returns:
            x_next: LongTensor [B, L] — resampled next state x_{t-1}.
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
        sample = copy_flag * x + (1 - copy_flag) * _x

        # Importance weight denominator: guidance multiplier at sampled tokens
        prob_multiplier = (
            (1 - copy_flag) * torch.gather(guidance.exp(), 2, _x.unsqueeze(-1)).squeeze(-1)
            + copy_flag * torch.ones_like(_x)
        )

        # Calculate exp(v_{t-1}(x_{t-1}) / alpha)
        expected_x0 = self.forward(sample, sigma_s)
        expected_x0_arg = torch.argmax(expected_x0, dim=2)
        expected_x0_onehot = torch.nn.functional.one_hot(
            expected_x0_arg,
            num_classes=self.vocab_size,
        )
        reward_num = reward_model(
            expected_x0_onehot.float().transpose(1, 2)[:, 0:4, :]
        ).detach()[:, 0][:, 0]

        # Calculate exp(v_t(x_t) / alpha)
        expected_x0 = self.forward(x, sigma_s)
        expected_x0_arg = torch.argmax(expected_x0, dim=2)
        expected_x0_onehot = torch.nn.functional.one_hot(
            expected_x0_arg,
            num_classes=self.vocab_size,
        )
        reward_den = reward_model(
            expected_x0_onehot.float().transpose(1, 2)[:, 0:4, :]
        ).detach()[:, 0][:, 0]

        # Set NaN values to 1 to avoid degenerate weights
        prob_multiplier[torch.isnan(prob_multiplier)] = 1
        ratio = (
            torch.exp(1.0 / alpha * (reward_num - reward_den))
            / prob_multiplier.prod(dim=-1)
        )
        ratio = ratio.detach().cpu().numpy()
        ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
        ratio_sum = ratio.sum()
        if ratio_sum <= 0:
            ratio = np.ones_like(ratio) / len(ratio)
        else:
            ratio = ratio / ratio_sum
        final_sample_indices = np.random.choice(
            reward_num.shape[0],
            reward_num.shape[0],
            p=ratio,
        )
        return sample[final_sample_indices]

    # ------------------------------------------------------------------
    # Sampling loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def controlled_sample_TDS(
        self,
        reward_model,
        alpha,
        guidance_scale,
        *,
        num_steps=None,
        eps=1e-5,
        eval_sp_size=None,
    ):
        """Generate samples using Twisted Diffusion Sampler guidance.

        Compatible with ``eval.py``'s
        ``sample_controlled(model, "TDS", reward_model, ...)``.

        Args:
            reward_model:   differentiable reward oracle.
            alpha:          float — SMC temperature.
            guidance_scale: float — CG gradient multiplier for the proposal.
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
                with torch.enable_grad():
                    x = self._ddpm_update_controlled_TDS(
                        x, t, dt, reward_model, alpha, guidance_scale
                    )
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
