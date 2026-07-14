"""Tests for miai_segmentation.models."""

import torch
from monai.networks.nets import UNet

from miai_segmentation.models import UNetConfig, build_unet

_TINY = UNetConfig(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(4, 8),
    strides=(2,),
    num_res_units=0,
)


def test_build_unet_returns_monai_unet() -> None:
    model = build_unet(_TINY)
    assert isinstance(model, UNet)


def test_build_unet_forward_shape_matches_input() -> None:
    model = build_unet(_TINY)
    x = torch.zeros(1, 1, 16, 16, 16)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 16, 16, 16)


def test_build_unet_default_config_is_valid() -> None:
    model = build_unet(UNetConfig())
    assert isinstance(model, UNet)
