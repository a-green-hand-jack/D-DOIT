"""
DiffusionSMC – Sequential Monte Carlo (SMC) controlled sampling for MDLM.

At each denoising step the base MDLM transition is applied, then particles are
resampled proportional to the incremental weight

    w ∝ exp( (v_{t-1}(x_{t-1}) − v_t(x_t)) / α )

where v_t(x_t) = r( E[x₀ | x_t] ) is the value function evaluated via a
one-step reward look-ahead.  No gradient computation is required.

References
----------
- diffusion_gosai_update.py  (original implementation in the Diffusion base class)
"""

import numpy as np
import torch

from ddoit.benchmarks.drakes_diffusion_gosai_update import (
    Diffusion,
    _sample_categorical,
)


class DiffusionSMC(Diffusion):
    """Diffusion model with Sequential Monte Carlo (SMC) guidance.

    Inherits the full MDLM diffusion model and overrides only the
    controlled-sampling methods.  Call via ``controlled_sample_SMC(...)`` —
    compatible with ``eval.py``'s ``sample_controlled(model, "SMC", ...)``.
    """

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _ddpm_update_controlled_SMC(self, x, t, dt, reward_model, alpha=1.0):
        """Single denoising step with SMC resampling.

        Args:
            x:            LongTensor [B, L] — current discrete state x_t.
            t:            Tensor [B, 1] — current timestep.
            dt:           float — timestep delta.
            reward_model: callable reward oracle accepting [B, 4, L] one-hot
                          float tensors, returning [B, num_targets];
                          column [:, 0, 0] is used.
            alpha:        float — temperature that controls resampling sharpness.

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
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        copy_flag = (x != self.mask_index).to(x.dtype)
        sample = copy_flag * x + (1 - copy_flag) * _sample_categorical(q_xs)

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

        ratio = torch.exp(1.0 / alpha * (reward_num - reward_den))
        ratio = ratio.detach().cpu().numpy()
        final_sample_indices = np.random.choice(
            reward_num.shape[0],
            reward_num.shape[0],
            p=ratio / ratio.sum(),
        )
        return sample[final_sample_indices]

    # ------------------------------------------------------------------
    # Sampling loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def controlled_sample_SMC(
        self,
        reward_model,
        alpha,
        *,
        num_steps=None,
        eps=1e-5,
        eval_sp_size=None,
    ):
        """Generate samples using Sequential Monte Carlo guidance.

        Compatible with ``eval.py``'s
        ``sample_controlled(model, "SMC", reward_model, ...)``.

        Args:
            reward_model: reward oracle — callable accepting [B, 4, L] float
                          one-hot tensors, returning [B, num_targets].
            alpha:        SMC temperature (controls resampling sharpness).
            num_steps:    total denoising steps (default from config).
            eps:          small constant to avoid t=0 (default 1e-5).
            eval_sp_size: batch size override (default from config).

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
                x = self._ddpm_update_controlled_SMC(x, t, dt, reward_model, alpha)
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
