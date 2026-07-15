"""Tests for miai_diffusion.schedule."""

import pytest
import torch

from miai_diffusion.exceptions import DiffusionError
from miai_diffusion.schedule import NoiseSchedule, NoiseScheduleConfig


class _ZeroNoiseModel(torch.nn.Module):
    """Predicts zero noise, for testing p_sample_step's arithmetic in isolation."""

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def test_schedule_betas_have_expected_length_and_range() -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(num_timesteps=10, beta_start=1e-4, beta_end=0.02))
    assert schedule.betas.shape == (10,)
    assert torch.all(schedule.betas >= 0)
    assert torch.all(schedule.betas <= 1)


def test_schedule_alpha_bars_are_monotonically_decreasing() -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(num_timesteps=20))
    diffs = schedule.alpha_bars[1:] - schedule.alpha_bars[:-1]
    assert torch.all(diffs <= 0)


def test_schedule_cosine_variant_builds_without_error() -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(num_timesteps=20, schedule="cosine"))
    assert schedule.betas.shape == (20,)


def test_schedule_unknown_variant_raises() -> None:
    config = NoiseScheduleConfig().model_copy(update={"schedule": "not_a_real_schedule"})
    with pytest.raises(DiffusionError):
        NoiseSchedule(config)


def test_q_sample_shape_matches_input() -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(num_timesteps=10))
    x0 = torch.zeros(2, 1, 4, 4, 4)
    t = torch.tensor([0, 5])
    noise = torch.randn_like(x0)

    x_t = schedule.q_sample(x0, t, noise)

    assert x_t.shape == x0.shape


def test_q_sample_at_t0_is_close_to_x0() -> None:
    schedule = NoiseSchedule(
        NoiseScheduleConfig(num_timesteps=1000, beta_start=1e-4, beta_end=0.02)
    )
    x0 = torch.ones(1, 1, 4, 4, 4)
    t = torch.tensor([0])
    noise = torch.zeros_like(x0)

    x_t = schedule.q_sample(x0, t, noise)

    assert torch.allclose(x_t, x0, atol=0.05)


def test_p_sample_step_output_shape_matches_input() -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(num_timesteps=10))
    model = _ZeroNoiseModel()
    x_t = torch.randn(2, 1, 4, 4, 4)

    x_prev = schedule.p_sample_step(model, x_t, 5)

    assert x_prev.shape == x_t.shape


def test_p_sample_step_final_step_is_deterministic() -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(num_timesteps=10))
    model = _ZeroNoiseModel()
    x_t = torch.randn(1, 1, 4, 4, 4)

    first = schedule.p_sample_step(model, x_t, 0)
    second = schedule.p_sample_step(model, x_t, 0)

    assert torch.allclose(first, second)


def test_p_sample_step_out_of_range_t_raises() -> None:
    schedule = NoiseSchedule(NoiseScheduleConfig(num_timesteps=10))
    model = _ZeroNoiseModel()
    x_t = torch.randn(1, 1, 4, 4, 4)

    with pytest.raises(DiffusionError):
        schedule.p_sample_step(model, x_t, 10)
