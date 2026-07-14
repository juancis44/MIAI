"""Segmentation model construction.

MIAI's reference segmentation task is binary 3D semantic segmentation,
built on :class:`monai.networks.nets.UNet`. Other architectures can be
added alongside :func:`build_unet` as the ecosystem grows, without
changing the training/inference APIs, which only depend on the model
being a standard :class:`torch.nn.Module`.
"""

from __future__ import annotations

from monai.networks.nets import UNet

from miai_core.config import MIAIBaseConfig


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
