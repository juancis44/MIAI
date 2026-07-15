"""A compact 3D UNet with timestep conditioning, for noise prediction."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn

from miai_core.config import MIAIBaseConfig


class DiffusionUNetConfig(MIAIBaseConfig):
    """Configuration for :func:`build_diffusion_unet`.

    Attributes:
        in_channels: Number of input image channels.
        base_channels: Number of channels at the finest resolution
            level.
        channel_multipliers: Per-level channel multiplier relative to
            ``base_channels``, finest-to-coarsest order (e.g. ``(1, 2,
            4)`` means three levels with ``base_channels``,
            ``2*base_channels``, and ``4*base_channels`` channels).
            Input spatial dimensions should be divisible by
            ``2 ** (len(channel_multipliers) - 1)`` so downsampling and
            the matching upsampling round-trip to the exact input size.
        time_embedding_dim: Dimensionality of the sinusoidal timestep
            embedding.
    """

    in_channels: int = 1
    base_channels: int = 16
    channel_multipliers: tuple[int, ...] = (1, 2, 4)
    time_embedding_dim: int = 64


class _SinusoidalTimeEmbedding(nn.Module):
    """Maps an integer timestep to a fixed sinusoidal embedding vector.

    Same construction as the positional encoding in "Attention Is All
    You Need" (Vaswani et al. 2017), applied to diffusion timesteps as
    in Ho, Jain & Abbeel 2020.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=t.device).float() / (half_dim - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class _ResBlock(nn.Module):
    """A pre-activation residual block, conditioned on a timestep embedding."""

    def __init__(self, in_channels: int, out_channels: int, time_embedding_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_embedding_dim, out_channels)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip: nn.Module = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(torch.relu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None, None]
        h = self.conv2(torch.relu(self.norm2(h)))
        return h + cast(torch.Tensor, self.skip(x))


class DiffusionUNet(nn.Module):
    """Predicts the noise added to a volume at a given diffusion timestep.

    A small 3D UNet: each resolution level is a residual block
    conditioned on a sinusoidal timestep embedding, connected by
    strided-convolution downsampling and transposed-convolution
    upsampling, with encoder/decoder skip connections -- the usual UNet
    pattern, adapted from Ho, Jain & Abbeel 2020's noise-prediction
    network.

    Call as ``model(x, t)`` where ``x`` is the noisy input, shape
    ``(B, in_channels, D, H, W)``, and ``t`` is the integer timestep per
    batch item, shape ``(B,)``. Returns the predicted noise, the same
    shape as ``x``.
    """

    def __init__(self, config: DiffusionUNetConfig) -> None:
        super().__init__()
        self.config = config
        time_dim = config.time_embedding_dim
        self.time_embedding = nn.Sequential(
            _SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        channels = [config.base_channels * m for m in config.channel_multipliers]

        self.in_conv = nn.Conv3d(config.in_channels, channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.down_blocks.append(_ResBlock(channels[i], channels[i], time_dim))
            self.downsamplers.append(
                nn.Conv3d(channels[i], channels[i + 1], kernel_size=4, stride=2, padding=1)
            )

        self.mid_block = _ResBlock(channels[-1], channels[-1], time_dim)

        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for i in reversed(range(len(channels) - 1)):
            self.upsamplers.append(
                nn.ConvTranspose3d(channels[i + 1], channels[i], kernel_size=4, stride=2, padding=1)
            )
            self.up_blocks.append(_ResBlock(channels[i] * 2, channels[i], time_dim))

        self.out_norm = nn.GroupNorm(min(8, channels[0]), channels[0])
        self.out_conv = nn.Conv3d(channels[0], config.in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(t)

        h = self.in_conv(x)
        skips = []
        for block, downsample in zip(self.down_blocks, self.downsamplers, strict=True):
            h = block(h, t_emb)
            skips.append(h)
            h = downsample(h)

        h = self.mid_block(h, t_emb)

        for block, upsample, skip in zip(
            self.up_blocks, self.upsamplers, reversed(skips), strict=True
        ):
            h = upsample(h)
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)

        return cast(torch.Tensor, self.out_conv(torch.relu(self.out_norm(h))))


def build_diffusion_unet(config: DiffusionUNetConfig) -> DiffusionUNet:
    """Build a :class:`DiffusionUNet` from a config.

    Args:
        config: The architecture configuration.

    Returns:
        An uninitialized (freshly constructed) diffusion UNet.
    """
    return DiffusionUNet(config)
