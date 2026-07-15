"""Forward (noising) and reverse (denoising) diffusion schedule.

Implements the standard DDPM forward process
``q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)`` and the
corresponding reverse (denoising) step, following Ho, Jain & Abbeel,
"Denoising Diffusion Probabilistic Models" (2020).
"""

from __future__ import annotations

import math
from typing import Literal

import torch

from miai_core.config import MIAIBaseConfig
from miai_diffusion.exceptions import DiffusionError


class NoiseScheduleConfig(MIAIBaseConfig):
    """Configuration for :class:`NoiseSchedule`.

    Attributes:
        num_timesteps: Number of diffusion steps ``T``.
        beta_start: Noise variance at ``t=0``.
        beta_end: Noise variance at ``t=T-1``.
        schedule: ``"linear"`` (Ho et al. 2020) or ``"cosine"``
            (Nichol & Dhariwal 2021).
    """

    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: Literal["linear", "cosine"] = "linear"


class NoiseSchedule:
    """Precomputed diffusion coefficients, and the forward/reverse steps that use them."""

    def __init__(self, config: NoiseScheduleConfig, device: torch.device | str = "cpu") -> None:
        """Precompute betas/alphas/alpha_bars for ``config`` on ``device``.

        Args:
            config: Schedule shape (number of steps, noise range, kind).
            device: Device the schedule's tensors live on -- should
                match the device the model and data are on.
        """
        self.config = config
        self.device = torch.device(device)
        self.betas = self._make_betas().to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def _make_betas(self) -> torch.Tensor:
        num_timesteps = self.config.num_timesteps
        if self.config.schedule == "linear":
            return torch.linspace(self.config.beta_start, self.config.beta_end, num_timesteps)
        if self.config.schedule == "cosine":
            steps = torch.arange(num_timesteps + 1, dtype=torch.float64) / num_timesteps
            offset = 0.008
            alpha_bar = torch.cos((steps + offset) / (1 + offset) * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            return torch.clip(betas, 1e-8, 0.999).float()
        raise DiffusionError(
            f"Unknown schedule '{self.config.schedule}'. Expected 'linear' or 'cosine'."
        )

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Sample ``x_t`` from ``x_0`` by adding noise per the forward process.

        Args:
            x0: Clean input, shape ``(B, C, ...)``.
            t: Timestep per batch item, shape ``(B,)``, values in
                ``[0, num_timesteps)``.
            noise: Standard normal noise, same shape as ``x0``.

        Returns:
            ``sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise``.
        """
        alpha_bar_t = self.alpha_bars[t].view(-1, *([1] * (x0.dim() - 1)))
        return alpha_bar_t.sqrt() * x0 + (1 - alpha_bar_t).sqrt() * noise

    def p_sample_step(self, model: torch.nn.Module, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """Reverse one diffusion step: predict and remove the noise added at step ``t``.

        Args:
            model: A trained noise-prediction model, callable as
                ``model(x, t_batch) -> predicted_noise`` (e.g.
                :class:`~miai_diffusion.model.DiffusionUNet`).
            x_t: The current noisy sample, shape ``(B, C, ...)``.
            t: The timestep ``x_t`` corresponds to (the same step for
                every item in the batch).

        Returns:
            ``x_{t-1}``: ``x_t``, one step less noisy.

        Raises:
            DiffusionError: If ``t`` is outside
                ``[0, num_timesteps)``.
        """
        if not 0 <= t < self.config.num_timesteps:
            raise DiffusionError(
                f"t={t} is outside the schedule's range [0, {self.config.num_timesteps})."
            )

        with torch.no_grad():
            batch_size = x_t.shape[0]
            t_batch = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)
            predicted_noise = model(x_t, t_batch)

            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]

            mean = (1 / alpha_t.sqrt()) * (
                x_t - (beta_t / (1 - alpha_bar_t).sqrt()) * predicted_noise
            )

            if t == 0:
                return mean

            noise = torch.randn_like(x_t)
            sigma_t = beta_t.sqrt()
            return mean + sigma_t * noise
