"""Tests for miai_segmentation.three_d.models."""

import pytest
import torch
from monai.networks.nets import SegResNet, UNet

from miai_segmentation.exceptions import SegmentationError
from miai_segmentation.three_d.models import (
    ArchitectureConfig,
    SegResNetConfig,
    UNetConfig,
    build_model,
    build_segresnet,
    build_unet,
)

_TINY_UNET = UNetConfig(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(4, 8),
    strides=(2,),
    num_res_units=0,
)

_TINY_SEGRESNET = SegResNetConfig(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    init_filters=8,
    blocks_down=(1, 1),
    blocks_up=(1,),
)


def test_build_unet_returns_monai_unet() -> None:
    model = build_unet(_TINY_UNET)
    assert isinstance(model, UNet)


def test_build_unet_forward_shape_matches_input() -> None:
    model = build_unet(_TINY_UNET)
    x = torch.zeros(1, 1, 16, 16, 16)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 16, 16, 16)


def test_build_unet_default_config_is_valid() -> None:
    model = build_unet(UNetConfig())
    assert isinstance(model, UNet)


def test_build_unet_with_dropout_forward_shape_matches_input() -> None:
    """dropout is a new UNetConfig field -- confirm it's actually wired
    through to MONAI's UNet (not just accepted and ignored) and the
    model still runs end to end."""
    config = UNetConfig(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(4, 8),
        strides=(2,),
        num_res_units=1,
        dropout=0.3,
    )
    model = build_unet(config)
    x = torch.zeros(1, 1, 16, 16, 16)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 16, 16, 16)


def test_build_segresnet_returns_monai_segresnet() -> None:
    model = build_segresnet(_TINY_SEGRESNET)
    assert isinstance(model, SegResNet)


def test_build_segresnet_forward_shape_matches_input() -> None:
    model = build_segresnet(_TINY_SEGRESNET)
    x = torch.zeros(1, 1, 16, 16, 16)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 16, 16, 16)


def test_build_segresnet_default_config_is_valid() -> None:
    model = build_segresnet(SegResNetConfig())
    assert isinstance(model, SegResNet)


def test_build_model_dispatches_to_unet() -> None:
    model = build_model(ArchitectureConfig(kind="unet", unet=_TINY_UNET))
    assert isinstance(model, UNet)


def test_build_model_dispatches_to_segresnet() -> None:
    model = build_model(ArchitectureConfig(kind="segresnet", segresnet=_TINY_SEGRESNET))
    assert isinstance(model, SegResNet)


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
        kind="not-a-real-architecture", unet=UNetConfig(), segresnet=SegResNetConfig()
    )
    with pytest.raises(SegmentationError):
        build_model(config)
