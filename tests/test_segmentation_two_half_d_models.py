"""Tests for miai_segmentation.two_half_d.models."""

import pytest
import torch
from monai.networks.nets import UNet

from miai_segmentation.exceptions import SegmentationError
from miai_segmentation.two_half_d.models import (
    ArchitectureConfig,
    StackedUNetConfig,
    build_model,
    build_stacked_unet,
)

_TINY_STACKED_UNET = StackedUNetConfig(
    spatial_dims=2,
    in_channels=3,
    out_channels=1,
    context_slices=3,
    channels=(4, 8),
    strides=(2,),
    num_res_units=0,
)


def test_build_stacked_unet_returns_monai_unet() -> None:
    model = build_stacked_unet(_TINY_STACKED_UNET)
    assert isinstance(model, UNet)


def test_build_stacked_unet_forward_shape_matches_stacked_input() -> None:
    model = build_stacked_unet(_TINY_STACKED_UNET)
    # 3 stacked adjacent slices in, 1-channel mask for the center slice out.
    x = torch.zeros(1, 3, 16, 16)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 16, 16)


def test_build_stacked_unet_default_config_is_valid() -> None:
    model = build_stacked_unet(StackedUNetConfig())
    assert isinstance(model, UNet)
    assert model.in_channels == StackedUNetConfig().context_slices


def test_build_model_dispatches_to_unet() -> None:
    model = build_model(ArchitectureConfig(kind="unet", unet=_TINY_STACKED_UNET))
    assert isinstance(model, UNet)


def test_build_model_default_config_dispatches_to_unet() -> None:
    model = build_model(ArchitectureConfig())
    assert isinstance(model, UNet)


def test_architecture_config_rejects_unknown_kind() -> None:
    with pytest.raises(Exception):  # noqa: B017 -- Pydantic's own ValidationError
        ArchitectureConfig(kind="not-a-real-architecture")  # type: ignore[arg-type]


def test_build_model_raises_on_bypassed_invalid_kind() -> None:
    # model_construct bypasses Pydantic validation -- exercises build_model's
    # own defensive check, not reachable through normal config loading.
    config = ArchitectureConfig.model_construct(
        kind="not-a-real-architecture", unet=StackedUNetConfig()
    )
    with pytest.raises(SegmentationError):
        build_model(config)
