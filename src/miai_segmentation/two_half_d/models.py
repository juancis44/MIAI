"""2.5D (stacked-adjacent-slice) segmentation model construction.

"2.5D" segments one target slice using spatial context from its
neighbors, without the memory/compute cost of a full 3D model: ``N``
adjacent slices (e.g. slice ``i-1``, ``i``, ``i+1``) are stacked along
the channel axis and fed to an otherwise-ordinary 2D architecture, which
predicts the mask for the center slice only. Architecturally this is
identical to :mod:`miai_segmentation.two_d`'s UNet -- the only
difference is ``in_channels`` equaling the stack size instead of ``1``
-- so :class:`StackedUNetConfig` is a :class:`~miai_segmentation.two_d.
models.UNetConfig` with that one field's default changed and a
``context_slices`` field documenting the convention. Building and
arranging the actual stacked-slice input tensors (which slices go in
which channel, edge-of-volume handling, etc.) is a data-loading concern
-- see `docs/user_guide.md` for how the rest of the pipeline stages
prepare per-case tensors when this modality is wired into a workflow.
"""

from __future__ import annotations

from typing import Literal

import torch
from monai.networks.nets import UNet

from miai_core.config import MIAIBaseConfig
from miai_segmentation.exceptions import SegmentationError


class StackedUNetConfig(MIAIBaseConfig):
    """Configuration for :func:`build_stacked_unet`.

    Mirrors :class:`monai.networks.nets.UNet`'s constructor arguments
    (see MONAI's documentation for the effect of each), with
    ``in_channels`` doubling as the number of stacked slices.

    Attributes:
        spatial_dims: Number of spatial dimensions (``2`` -- the model
            still convolves over a single slice's height/width; the
            adjacent-slice context lives in the channel axis).
        in_channels: Number of stacked adjacent slices fed as input
            channels. ``context_slices`` should match this.
        out_channels: Number of output segmentation channels for the
            center slice's mask. ``1`` for binary segmentation.
        context_slices: Documents how many adjacent slices
            ``in_channels`` represents (e.g. ``3`` for one slice on
            each side of the target). Informational only -- not passed
            to MONAI -- but keeps the stacking convention explicit
            alongside the channel count it must match.
        channels: Number of output channels for each encoder/decoder
            resolution level.
        strides: Downsampling stride between consecutive levels in
            ``channels``; must have one fewer entry than ``channels``.
        num_res_units: Number of residual units per level.
    """

    spatial_dims: int = 2
    in_channels: int = 3
    out_channels: int = 1
    context_slices: int = 3
    channels: tuple[int, ...] = (16, 32, 64, 128)
    strides: tuple[int, ...] = (2, 2, 2)
    num_res_units: int = 2


def build_stacked_unet(config: StackedUNetConfig) -> UNet:
    """Build a :class:`monai.networks.nets.UNet` from a config.

    Args:
        config: The architecture configuration.

    Returns:
        An uninitialized (freshly constructed) MONAI UNet with
        ``in_channels`` set for stacked-slice input.
    """
    return UNet(
        spatial_dims=config.spatial_dims,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        channels=config.channels,
        strides=config.strides,
        num_res_units=config.num_res_units,
    )


class ArchitectureConfig(MIAIBaseConfig):
    """Selects and configures one 2.5D segmentation architecture.

    Only one architecture exists today (:class:`StackedUNetConfig`), but
    this keeps the same ``ArchitectureConfig``/``build_model`` shape as
    :mod:`miai_segmentation.three_d` and :mod:`miai_segmentation.two_d`,
    so a caller (or a future pipeline stage) can depend on one consistent
    entry point across modalities. Extend ``kind`` when a second 2.5D
    architecture is added -- do not replace ``build_model`` with a bare
    ``build_stacked_unet`` call at call sites, or every caller needs to
    change when that happens.

    Attributes:
        kind: Which architecture :func:`build_model` constructs.
        unet: Used when ``kind == "unet"``.
    """

    kind: Literal["unet"] = "unet"
    unet: StackedUNetConfig = StackedUNetConfig()


def build_model(config: ArchitectureConfig) -> torch.nn.Module:
    """Build the 2.5D architecture selected by ``config.kind``.

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
        return build_stacked_unet(config.unet)
    raise SegmentationError(f"Unknown 2.5D architecture kind: {config.kind!r}")
