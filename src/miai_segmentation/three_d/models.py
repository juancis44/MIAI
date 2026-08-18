"""3D segmentation model construction.

MIAI's 3D segmentation modality operates on full volumes (``spatial_dims
= 3``). Two representative architectures are provided: :func:`build_unet`
(:class:`monai.networks.nets.UNet`, an encoder/decoder with residual
units -- MIAI's original reference model) and :func:`build_segresnet`
(:class:`monai.networks.nets.SegResNet`, the residual-block encoder/decoder
from Myronenko 2018's BraTS-winning design, a common comparison point for
3D volumetric segmentation). :func:`build_model` dispatches between them
from a single :class:`ArchitectureConfig`, so pipeline stages only need
to depend on one config/build entry point regardless of which 3D
architecture an experiment picks.
"""

from __future__ import annotations

from typing import Literal

import torch
from monai.networks.nets import SegResNet, UNet

from miai_core.config import MIAIBaseConfig
from miai_segmentation.exceptions import SegmentationError


class UNetConfig(MIAIBaseConfig):
    """Configuration for :func:`build_unet`.

    Mirrors :class:`monai.networks.nets.UNet`'s constructor arguments;
    see MONAI's documentation for the effect of each.

    Attributes:
        spatial_dims: Number of spatial dimensions (``3`` for volumes).
        in_channels: Number of input image channels.
        out_channels: Number of output segmentation channels. ``1`` for
            binary segmentation (foreground vs. background).
        channels: Number of output channels for each encoder/decoder
            resolution level.
        strides: Downsampling stride between consecutive levels in
            ``channels``; must have one fewer entry than ``channels``.
        num_res_units: Number of residual units per level.
    """

    spatial_dims: int = 3
    in_channels: int = 1
    out_channels: int = 1
    channels: tuple[int, ...] = (16, 32, 64, 128)
    strides: tuple[int, ...] = (2, 2, 2)
    num_res_units: int = 2


def build_unet(config: UNetConfig) -> UNet:
    """Build a :class:`monai.networks.nets.UNet` from a config.

    Args:
        config: The architecture configuration.

    Returns:
        An uninitialized (freshly constructed) MONAI UNet.
    """
    return UNet(
        spatial_dims=config.spatial_dims,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        channels=config.channels,
        strides=config.strides,
        num_res_units=config.num_res_units,
    )


class SegResNetConfig(MIAIBaseConfig):
    """Configuration for :func:`build_segresnet`.

    Mirrors the subset of :class:`monai.networks.nets.SegResNet`'s
    constructor arguments relevant to MIAI's binary-segmentation use
    case; see MONAI's documentation for the effect of each.

    Attributes:
        spatial_dims: Number of spatial dimensions (``3`` for volumes).
        in_channels: Number of input image channels.
        out_channels: Number of output segmentation channels. ``1`` for
            binary segmentation (foreground vs. background), matching
            :class:`UNetConfig`'s convention (MONAI's own default is
            ``2``, for multi-class use).
        init_filters: Number of output channels of the first
            convolution layer; doubles at each downsampling level.
        dropout_prob: Dropout probability applied within each residual
            block. ``None`` disables dropout.
        blocks_down: Number of residual blocks at each encoder
            resolution level.
        blocks_up: Number of residual blocks at each decoder
            resolution level; has one fewer entry than ``blocks_down``.
    """

    spatial_dims: int = 3
    in_channels: int = 1
    out_channels: int = 1
    init_filters: int = 8
    dropout_prob: float | None = None
    blocks_down: tuple[int, ...] = (1, 2, 2, 4)
    blocks_up: tuple[int, ...] = (1, 1, 1)


def build_segresnet(config: SegResNetConfig) -> SegResNet:
    """Build a :class:`monai.networks.nets.SegResNet` from a config.

    Args:
        config: The architecture configuration.

    Returns:
        An uninitialized (freshly constructed) MONAI SegResNet.
    """
    return SegResNet(
        spatial_dims=config.spatial_dims,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        init_filters=config.init_filters,
        dropout_prob=config.dropout_prob,
        blocks_down=config.blocks_down,
        blocks_up=config.blocks_up,
    )


class ArchitectureConfig(MIAIBaseConfig):
    """Selects and configures one 3D segmentation architecture.

    A single config a pipeline stage can depend on regardless of which
    3D architecture an experiment uses -- only ``kind`` and the matching
    nested config need to be set; the other nested config is ignored.

    Attributes:
        kind: Which architecture :func:`build_model` constructs.
        unet: Used when ``kind == "unet"``.
        segresnet: Used when ``kind == "segresnet"``.
    """

    kind: Literal["unet", "segresnet"] = "unet"
    unet: UNetConfig = UNetConfig()
    segresnet: SegResNetConfig = SegResNetConfig()


def build_model(config: ArchitectureConfig) -> torch.nn.Module:
    """Build the 3D architecture selected by ``config.kind``.

    Args:
        config: The architecture selection and its per-kind settings.

    Returns:
        An uninitialized (freshly constructed) model for the selected
        architecture.

    Raises:
        SegmentationError: If ``config.kind`` is not a recognized
            architecture. In practice this cannot happen through normal
            use -- ``kind`` is a :class:`typing.Literal` validated by
            Pydantic -- but it guards against a config bypassing that
            validation (e.g. constructed via ``model_construct``).
    """
    if config.kind == "unet":
        return build_unet(config.unet)
    if config.kind == "segresnet":
        return build_segresnet(config.segresnet)
    raise SegmentationError(f"Unknown 3D architecture kind: {config.kind!r}")
