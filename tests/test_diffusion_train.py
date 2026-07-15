"""Tests for miai_diffusion.train (tiny real tensors, CPU only)."""

from pathlib import Path

import pytest
import torch

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
