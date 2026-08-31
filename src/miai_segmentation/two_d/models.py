"""2D segmentation model construction.

MIAI's 2D segmentation modality operates per-slice (``spatial_dims =
2``), for cases where a full 3D volume is unnecessary or too costly to
process at once. Three representative architectures are provided:
:func:`build_unet` (:class:`monai.networks.nets.UNet`, the same
residual-unit encoder/decoder used by
:mod:`miai_segmentation.three_d`, here at ``spatial_dims=2``),
:func:`build_attention_unet` (:class:`monai.networks.nets.AttentionUnet`,
Oktay et al. 2018 -- adds attention gates on the skip connections so the
decoder can down-weight irrelevant encoder features, a common choice
when the foreground occupies a small fraction of each slice), and
:func:`build_res_attention_unet` (:class:`ResAttentionUNet`, this
module's own -- combines the first two: a residual-block
encoder/decoder (like ``build_unet``'s) with attention-gated skip
connections (like ``build_attention_unet``'s), a combination neither
MONAI nor MIAI previously offered as a single architecture).
``ResAttentionUnetConfig.use_attention=False`` builds the same class
with its attention gates removed -- a plain residual U-Net with
ordinary skip connections, letting the two be compared with everything
else (channel depth/width, dropout) held identical.
:func:`build_model` dispatches between all three from a single
:class:`ArchitectureConfig`, mirroring
:mod:`miai_segmentation.three_d.models`'s pattern.
"""

from __future__ import annotations

from typing import Literal

import torch
from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.nets import AttentionUnet, UNet
from torch import nn

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


class _AttentionGate(nn.Module):
    """Additive attention gate on a skip connection (Oktay et al. 2018).

    Learns a per-pixel gating coefficient in ``[0, 1]`` from the
    decoder's up-sampled feature map (the gating signal, ``gate``) and
    the encoder's skip feature map (``skip``), then scales ``skip`` by
    it before concatenation -- down-weighting encoder regions the
    decoder's current context finds irrelevant, rather than passing the
    full encoder features through unchanged like a plain UNet's skip
    connection does. Implemented directly on
    :class:`~monai.networks.blocks.Convolution` rather than importing
    from ``monai.networks.nets.attentionunet``, whose own
    ``AttentionBlock`` is private (not part of MONAI's public API).
    """

    def __init__(
        self,
        spatial_dims: int,
        gate_channels: int,
        skip_channels: int,
        inter_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.gate_conv = Convolution(
            spatial_dims, gate_channels, inter_channels, kernel_size=1, padding=0, conv_only=True
        )
        self.skip_conv = Convolution(
            spatial_dims, skip_channels, inter_channels, kernel_size=1, padding=0, conv_only=True
        )
        self.psi = Convolution(
            spatial_dims, inter_channels, 1, kernel_size=1, padding=0, conv_only=True
        )
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        combined = self.relu(self.gate_conv(gate) + self.skip_conv(skip))
        attention = self.sigmoid(self.psi(combined))
        result: torch.Tensor = self.dropout(skip * attention)
        return result


class ResAttentionUNet(nn.Module):
    """Residual-block U-Net with attention-gated skip connections.

    Combines two ideas MIAI's other two 2D architectures each
    contribute one of: :class:`UNetConfig`'s residual encoder/decoder
    blocks (:class:`monai.networks.blocks.ResidualUnit`, the same block
    :class:`monai.networks.nets.UNet` builds on) and
    :class:`AttentionUnetConfig`'s attention-gated skip connections
    (:class:`_AttentionGate`, above). Built directly on MONAI's public
    building blocks rather than as a subclass of either MONAI network.

    Structurally: an encoder of ``len(channels)`` :class:`ResidualUnit`
    levels (the first at stride 1, each subsequent one downsampling by
    the matching entry in ``strides``), then a mirrored decoder where
    each level transposed-convolves back up, attention-gates the
    matching encoder skip connection using the up-sampled features as
    the gating signal, concatenates the two, and fuses them with
    another :class:`ResidualUnit` -- finished with a ``1x1`` convolution
    to ``out_channels``.

    Each attention gate's bottleneck width (``inter_channels``, the two
    ``1x1`` gate/skip projections' shared output width before the final
    projection to a single-channel coefficient map) is
    ``max(up_out // attention_reduction, 1)`` -- a narrower bottleneck
    (larger ``attention_reduction``) compresses the gate's decision
    into fewer channels, a wider one (smaller ``attention_reduction``,
    down to ``1`` for no compression at all) gives it more capacity to
    make a nuanced per-pixel decision at the cost of extra parameters.

    Setting ``use_attention=False`` removes the attention gates
    entirely -- each skip connection is concatenated unmodified, as a
    plain (non-attention-gated) residual U-Net would -- while keeping
    every other structural choice (residual encoder/decoder blocks,
    channel depth/width, dropout) identical. This isolates whether the
    attention mechanism itself, versus the residual-block architecture
    it sits on, is responsible for a given result: the same class and
    forward-pass structure either way, with only the gating step
    present or absent.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: tuple[int, ...],
        strides: tuple[int, ...],
        num_res_units: int = 2,
        dropout: float = 0.0,
        attention_reduction: int = 2,
        use_attention: bool = True,
    ) -> None:
        """Build the encoder/decoder residual blocks and attention gates."""
        super().__init__()
        if len(strides) != len(channels) - 1:
            raise SegmentationError(
                "strides must have exactly one fewer entry than channels: got "
                f"{len(channels)} channels and {len(strides)} strides."
            )
        if attention_reduction < 1:
            raise SegmentationError(f"attention_reduction must be >= 1: got {attention_reduction}.")

        self.use_attention = use_attention

        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for level, level_channels in enumerate(channels):
            stride = 1 if level == 0 else strides[level - 1]
            self.encoders.append(
                ResidualUnit(
                    spatial_dims,
                    prev_channels,
                    level_channels,
                    strides=stride,
                    subunits=num_res_units,
                    dropout=dropout,
                )
            )
            prev_channels = level_channels

        self.up_convs = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(len(channels) - 1, 0, -1):
            up_in, up_out, stride = channels[level], channels[level - 1], strides[level - 1]
            self.up_convs.append(
                Convolution(
                    spatial_dims,
                    up_in,
                    up_out,
                    strides=stride,
                    kernel_size=3,
                    is_transposed=True,
                    dropout=dropout,
                )
            )
            if use_attention:
                self.attention_gates.append(
                    _AttentionGate(
                        spatial_dims,
                        gate_channels=up_out,
                        skip_channels=up_out,
                        inter_channels=max(up_out // attention_reduction, 1),
                        dropout=dropout,
                    )
                )
            self.decoders.append(
                ResidualUnit(
                    spatial_dims,
                    up_out * 2,
                    up_out,
                    strides=1,
                    subunits=num_res_units,
                    dropout=dropout,
                )
            )

        self.final_conv = Convolution(
            spatial_dims, channels[0], out_channels, kernel_size=1, padding=0, conv_only=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual encoder, then the (optionally attention-gated) decoder."""
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
        skips.pop()  # the bottleneck's own output is not gated against itself

        for i, (up_conv, decoder) in enumerate(zip(self.up_convs, self.decoders, strict=True)):
            x = up_conv(x)
            skip = skips.pop()
            if self.use_attention:
                skip = self.attention_gates[i](gate=x, skip=skip)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        result: torch.Tensor = self.final_conv(x)
        return result


class ResAttentionUnetConfig(MIAIBaseConfig):
    """Configuration for :func:`build_res_attention_unet`.

    Same shape as :class:`UNetConfig` -- see its docstring for each
    field's effect; ``num_res_units`` and ``dropout`` apply to both the
    encoder's and the decoder's residual blocks (the attention gates
    have no separate dropout knob beyond this ``dropout`` value).

    Attributes:
        spatial_dims: Number of spatial dimensions (``2`` for slices).
        in_channels: Number of input image channels.
        out_channels: Number of output segmentation channels.
        channels: Number of output channels for each encoder/decoder
            resolution level.
        strides: Downsampling stride between consecutive levels in
            ``channels``; must have one fewer entry than ``channels``.
        num_res_units: Number of convolutions per residual block
            (:class:`monai.networks.blocks.ResidualUnit`'s
            ``subunits``).
        dropout: Dropout probability applied within each residual
            block and each attention gate.
        attention_reduction: Divisor applied to each attention gate's
            skip-connection channel count to get its bottleneck width
            (``max(up_out // attention_reduction, 1)``). ``2`` (the
            default, unchanged from this field's introduction) matches
            the common Attention U-Net convention of halving; ``1``
            disables the compression entirely (the gate's bottleneck is
            as wide as the skip connection itself). Ignored when
            ``use_attention`` is ``False``.
        use_attention: Whether to attention-gate each skip connection
            at all. ``True`` (the default, unchanged from this field's
            introduction) preserves every existing behavior; ``False``
            builds a plain residual U-Net with the same encoder/decoder
            structure but ordinary (ungated) skip connections -- see
            :class:`ResAttentionUNet`'s docstring.
    """

    spatial_dims: int = 2
    in_channels: int = 1
    out_channels: int = 1
    channels: tuple[int, ...] = (16, 32, 64, 128)
    strides: tuple[int, ...] = (2, 2, 2)
    num_res_units: int = 2
    dropout: float = 0.0
    attention_reduction: int = 2
    use_attention: bool = True


def build_res_attention_unet(config: ResAttentionUnetConfig) -> ResAttentionUNet:
    """Build a :class:`ResAttentionUNet` from a config.

    Args:
        config: The architecture configuration.

    Returns:
        An uninitialized (freshly constructed) residual, attention-gated
        UNet.
    """
    return ResAttentionUNet(
        spatial_dims=config.spatial_dims,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        channels=config.channels,
        strides=config.strides,
        num_res_units=config.num_res_units,
        dropout=config.dropout,
        attention_reduction=config.attention_reduction,
        use_attention=config.use_attention,
    )


class ArchitectureConfig(MIAIBaseConfig):
    """Selects and configures one 2D segmentation architecture.

    A single config a pipeline stage can depend on regardless of which
    2D architecture an experiment uses -- only ``kind`` and the matching
    nested config need to be set; the other nested configs are ignored.

    Attributes:
        kind: Which architecture :func:`build_model` constructs.
        unet: Used when ``kind == "unet"``.
        attention_unet: Used when ``kind == "attention_unet"``.
        res_attention_unet: Used when ``kind == "res_attention_unet"``.
    """

    kind: Literal["unet", "attention_unet", "res_attention_unet"] = "unet"
    unet: UNetConfig = UNetConfig()
    attention_unet: AttentionUnetConfig = AttentionUnetConfig()
    res_attention_unet: ResAttentionUnetConfig = ResAttentionUnetConfig()


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
    if config.kind == "res_attention_unet":
        return build_res_attention_unet(config.res_attention_unet)
    raise SegmentationError(f"Unknown 2D architecture kind: {config.kind!r}")
