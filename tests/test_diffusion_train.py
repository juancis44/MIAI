"""Tests for miai_diffusion.train (tiny real tensors, CPU only)."""

import copy
from pathlib import Path

import pytest
import torch
from torch.nn import functional

from miai_diffusion.exceptions import DiffusionError
from miai_diffusion.model import DiffusionUNetConfig, build_diffusion_unet
from miai_diffusion.schedule import NoiseSchedule, NoiseScheduleConfig
from miai_diffusion.train import DiffusionTrainingConfig, train_diffusion_model

_UNET_CONFIG = DiffusionUNetConfig(
    in_channels=1, base_channels=4, channel_multipliers=(1, 2), time_embedding_dim=16
)


def _fake_loader(n_batches: int, batch_size: int = 1) -> list[dict[str, torch.Tensor]]:
    return [{"image": torch.randn(batch_size, 1, 8, 8, 8)} for _ in range(n_batches)]


@pytest.mark.slow
def test_train_diffusion_model_writes_checkpoint(tmp_path: Path) -> None:
    model = build_diffusion_unet(_UNET_CONFIG)
    schedule = NoiseSchedule(NoiseScheduleConfig(num_timesteps=50), device="cpu")
    config = DiffusionTrainingConfig(max_epochs=1, device="cpu")

    checkpoint_path = train_diffusion_model(
        model, _fake_loader(2), schedule, config, str(tmp_path / "checkpoints")
    )

    assert checkpoint_path.exists()
    assert checkpoint_path.name == config.checkpoint_name

    fresh_model = build_diffusion_unet(_UNET_CONFIG)
    fresh_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))


def test_train_diffusion_model_empty_loader_raises(tmp_path: Path) -> None:
    model = build_diffusion_unet(_UNET_CONFIG)
    schedule = NoiseSchedule(NoiseScheduleConfig(num_timesteps=50), device="cpu")
    config = DiffusionTrainingConfig(max_epochs=1, device="cpu")

    with pytest.raises(DiffusionError):
        train_diffusion_model(model, [], schedule, config, str(tmp_path / "unused"))


def _fixed_pattern_loader(n_batches: int, batch_size: int = 1) -> list[dict[str, torch.Tensor]]:
    """A loader with a single, consistent learnable signal (a centered
    cube), repeated across every batch -- unlike _fake_loader, which
    yields fresh random noise per batch and therefore has no signal a
    model could learn to denoise.
    """
    pattern = torch.zeros(1, 1, 8, 8, 8)
    pattern[:, :, 2:6, 2:6, 2:6] = 1.0
    return [pattern.repeat(batch_size, 1, 1, 1, 1).clone() for _ in range(n_batches)]


@pytest.mark.slow
def test_train_diffusion_model_actually_learns(tmp_path: Path) -> None:
    """Trains on a fixed, repeated pattern and checks the noise-prediction
    loss on a fixed evaluation sample drops relative to a freshly
    initialized model of the same architecture -- unlike
    test_train_diffusion_model_writes_checkpoint, which only checks
    training runs without error and produces a loadable checkpoint,
    never that the model actually learned to predict noise better.
    """
    model = build_diffusion_unet(_UNET_CONFIG)
    initial_state = copy.deepcopy(model.state_dict())
    schedule = NoiseSchedule(NoiseScheduleConfig(num_timesteps=50), device="cpu")
    config = DiffusionTrainingConfig(max_epochs=60, learning_rate=1e-2, device="cpu")

    checkpoint_path = train_diffusion_model(
        model, _fixed_pattern_loader(4), schedule, config, str(tmp_path / "checkpoints")
    )

    trained_model = build_diffusion_unet(_UNET_CONFIG)
    trained_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    untrained_model = build_diffusion_unet(_UNET_CONFIG)
    untrained_model.load_state_dict(initial_state)

    x0 = _fixed_pattern_loader(1)[0]
    generator = torch.Generator().manual_seed(123)
    t = torch.randint(0, schedule.config.num_timesteps, (x0.shape[0],), generator=generator)
    noise = torch.randn(x0.shape, generator=generator)
    x_t = schedule.q_sample(x0, t, noise)

    trained_model.eval()
    untrained_model.eval()
    with torch.no_grad():
        trained_loss = functional.mse_loss(trained_model(x_t, t), noise).item()
        untrained_loss = functional.mse_loss(untrained_model(x_t, t), noise).item()

    assert trained_loss < untrained_loss
