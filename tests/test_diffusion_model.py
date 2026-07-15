"""Tests for miai_diffusion.model."""

import torch

from miai_diffusion.model import DiffusionUNetConfig, build_diffusion_unet

_TWO_LEVEL = DiffusionUNetConfig(
    in_channels=1, base_channels=4, channel_multipliers=(1, 2), time_embedding_dim=16
)
_THREE_LEVEL = DiffusionUNetConfig(
    in_channels=1, base_channels=4, channel_multipliers=(1, 2, 4), time_embedding_dim=16
)


def test_diffusion_unet_two_level_output_shape_matches_input() -> None:
    model = build_diffusion_unet(_TWO_LEVEL)
    x = torch.randn(2, 1, 8, 8, 8)
    t = torch.tensor([0, 5])

    out = model(x, t)

    assert out.shape == x.shape


def test_diffusion_unet_three_level_output_shape_matches_input() -> None:
    model = build_diffusion_unet(_THREE_LEVEL)
    x = torch.randn(2, 1, 8, 8, 8)
    t = torch.tensor([3, 7])

    out = model(x, t)

    assert out.shape == x.shape


def test_diffusion_unet_different_timesteps_give_different_output() -> None:
    model = build_diffusion_unet(_TWO_LEVEL)
    model.eval()
    x = torch.randn(1, 1, 8, 8, 8)

    with torch.no_grad():
        out_t0 = model(x, torch.tensor([0]))
        out_t5 = model(x, torch.tensor([5]))

    assert not torch.allclose(out_t0, out_t5)
