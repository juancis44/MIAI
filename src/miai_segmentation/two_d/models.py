"""2D segmentation model construction.

MIAI's 2D segmentation modality operates per-slice (``spatial_dims =
2``), for cases where a full 3D volume is unnecessary or too costly to
process at once. Two representative architectures are provided:
:func:`build_unet` (:class:`monai.networks.nets.UNet`, the same
residual-unit encoder/decoder used by
:mod:`miai_segmentation.three_d`, here at ``spatial_dims=2``) and
:func:`build_attention_unet` (:class:`monai.networks.nets.AttentionUnet`,
Oktay et al. 2018 -- adds attention gates on the skip connections so the
decoder can down-weight irrelevant encoder features, a common choice
when the foreground occupies a small fraction of each slice).
:func:`build_model` dispatches between them from a single
:class:`ArchitectureConfig`, mirroring
:mod:`miai_segmentation.three_d.models`'s pattern.
"""

from __future__ import annotations

from typing import Literal

import torch
from monai.networks.nets import AttentionUnet, UNet

from miai_core.config import MIAIBaseConfig
from miai_segmentation.exceptions import SegmentationError


class UNetConfig(MIAIBaseConfig):
    """Configuration for :func:`build_unet`.

    Mirrors :class:`monai.networks.nets.UNet`'s constructor arguments;
    see MONAI's documentation for the effect of each. Structurally the
    same shape as :class:`miai_segmentation.three_d.models.UNetConfig`,
    defaulted to ``spatial_dims=2`` -- kept as its own class (rather than
    imported from :mod:`miai_segmentation.three_d`) so this modality's
    public API is self-contained, per `docs/api_design.md`'s "Package
    public surface" section.

    Attributes:
        spatial_dims: Number of spatial dimensions (``2`` for slices).
        in_channels: Number of input image channels.
        out_channels: Number of output segmentation channels. ``1`` for
            binary segmentation (foreground vs. background).
        channels: Number of output channels for each encoder/decoder
            resolution level.
        strides: Downsampling stride between consecutive levels in
            ``channels``; must have one fewer entry than ``channels``.
        num_res_units: Number of residual units per level.
        dropout: Dropout probability applied within each residual
            unit's ADN block. ``0.0`` (the default) disables dropout,
            unchanged from this config's original behavior -- matches
            :class:`monai.networks.nets.UNet`'s own default.
    """

    spatial_dims: int = 2
    in_channels: int = 1
    out_channels: int = 1
    channels: tuple[int, ...] = (16, 32, 64, 128)
    strides: tuple[int, ...] = (2, 2, 2)
    num_res_units: int = 2
    dropout: float = 0.0


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
        dropout=config.dropout,
    )


class AttentionUnetConfig(MIAIBaseConfig):
    """Configuration for :func:`build_attention_unet`.

    Mirrors :class:`monai.networks.nets.AttentionUnet`'s constructor
    arguments; see MONAI's documentation for the effect of each.

    Attributes:
        spatial_dims: Number of spatial dimensions (``2`` for slices).
        in_channels: Number of input image channels.
        out_channels: Number of output segmentation channels. ``1`` for
            binary segmentation, matching :class:`UNetConfig`'s
            convention.
        channels: Number of output channels for each encoder/decoder
            resolution level.
        strides: Downsampling stride between consecutive levels in
            ``channels``; must have one fewer entry than ``channels``.
        dropout: Dropout probability applied within the network.
    """

    spatial_dims: int = 2
    in_channels: int = 1
    out_channels: int = 1
    channels: tuple[int, ...] = (16, 32, 64, 128)
    strides: tuple[int, ...] = (2, 2, 2)
    dropout: float = 0.0


def build_attention_unet(config: AttentionUnetConfig) -> AttentionUnet:
    """Build a :class:`monai.networks.nets.AttentionUnet` from a config.

    Args:
        config: The architecture configuration.

    Returns:
        An uninitialized (freshly constructed) MONAI AttentionUnet.
    """
    return AttentionUnet(
        spatial_dims=config.spatial_dims,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        channels=config.channels,
        strides=config.strides,
        dropout=config.dropout,
    )


class ArchitectureConfig(MIAIBaseConfig):
    """Selects and configures one 2D segmentation architecture.

    A single config a pipeline stage can depend on regardless of which
    2D architecture an experiment uses -- only ``kind`` and the matching
    nested config need to be set; the other nested config is ignored.

    Attributes:
        kind: Which architecture :func:`build_model` constructs.
        unet: Used when ``kind == "unet"``.
        attention_unet: Used when ``kind == "attention_unet"``.
    """

    kind: Literal["unet", "attention_unet"] = "unet"
    unet: UNetConfig = UNetConfig()
    attention_unet: AttentionUnetConfig = AttentionUnetConfig()


def build_model(config: ArchitectureConfig) -> torch.nn.Module:
    """Build the 2D architecture selected by ``config.kind``.

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
    if config.kind == "attention_unet":
        return build_attention_unet(config.attention_unet)
    raise SegmentationError(f"Unknown 2D architecture kind: {config.kind!r}")
