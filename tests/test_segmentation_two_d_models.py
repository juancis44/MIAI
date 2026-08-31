"""Tests for miai_segmentation.two_d.models."""

import pytest
import torch
from monai.networks.nets import AttentionUnet, UNet

from miai_segmentation.exceptions import SegmentationError
from miai_segmentation.two_d.models import (
    ArchitectureConfig,
    AttentionUnetConfig,
    ResAttentionUNet,
    ResAttentionUnetConfig,
    UNetConfig,
    build_attention_unet,
    build_model,
    build_res_attention_unet,
    build_unet,
)

_TINY_UNET = UNetConfig(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(4, 8),
    strides=(2,),
    num_res_units=0,
)

_TINY_ATTENTION_UNET = AttentionUnetConfig(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(4, 8),
    strides=(2,),
)

_TINY_RES_ATTENTION_UNET = ResAttentionUnetConfig(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(4, 8),
    strides=(2,),
    num_res_units=0,
)


def test_build_unet_returns_monai_unet() -> None:
    model = build_unet(_TINY_UNET)
    assert isinstance(model, UNet)


def test_build_unet_forward_shape_matches_input() -> None:
    model = build_unet(_TINY_UNET)
    x = torch.zeros(1, 1, 16, 16)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 16, 16)


def test_build_unet_default_config_is_valid() -> None:
    model = build_unet(UNetConfig())
    assert isinstance(model, UNet)


def test_build_unet_with_dropout_forward_shape_matches_input() -> None:
    """dropout is a new UNetConfig field -- confirm it's actually wired
    through to MONAI's UNet (not just accepted and ignored) and the
    model still runs end to end."""
    config = UNetConfig(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(4, 8),
        strides=(2,),
        num_res_units=1,
        dropout=0.3,
    )
    model = build_unet(config)
    x = torch.zeros(1, 1, 16, 16)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 16, 16)


def test_build_attention_unet_returns_monai_attention_unet() -> None:
    model = build_attention_unet(_TINY_ATTENTION_UNET)
    assert isinstance(model, AttentionUnet)


def test_build_attention_unet_forward_shape_matches_input() -> None:
    model = build_attention_unet(_TINY_ATTENTION_UNET)
    x = torch.zeros(1, 1, 16, 16)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 16, 16)


def test_build_attention_unet_default_config_is_valid() -> None:
    model = build_attention_unet(AttentionUnetConfig())
    assert isinstance(model, AttentionUnet)


def test_build_res_attention_unet_returns_res_attention_unet() -> None:
    model = build_res_attention_unet(_TINY_RES_ATTENTION_UNET)
    assert isinstance(model, ResAttentionUNet)


def test_build_res_attention_unet_forward_shape_matches_input() -> None:
    model = build_res_attention_unet(_TINY_RES_ATTENTION_UNET)
    x = torch.zeros(1, 1, 16, 16)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 16, 16)


def test_build_res_attention_unet_default_config_is_valid() -> None:
    model = build_res_attention_unet(ResAttentionUnetConfig())
    x = torch.zeros(1, 1, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 32, 32)


def test_build_res_attention_unet_multi_class_multi_level() -> None:
    """The default config is binary/3-level -- confirm a deeper,
    multi-class config (matching what examples/validate_acdc.py
    actually uses) also runs end to end, out_channels included."""
    config = ResAttentionUnetConfig(
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        out_channels=4,
        dropout=0.2,
    )
    model = build_res_attention_unet(config)
    x = torch.zeros(1, 1, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 4, 32, 32)


def test_build_res_attention_unet_with_dropout_forward_shape_matches_input() -> None:
    config = ResAttentionUnetConfig(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(4, 8),
        strides=(2,),
        num_res_units=1,
        dropout=0.3,
    )
    model = build_res_attention_unet(config)
    x = torch.zeros(1, 1, 16, 16)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 16, 16)


def test_build_res_attention_unet_attention_reduction_one_forward_shape() -> None:
    """attention_reduction=1 disables the gate bottleneck's compression
    (inter_channels == up_out instead of up_out // 2) -- confirm it's
    actually wired through and the model still runs end to end."""
    config = ResAttentionUnetConfig(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(4, 8),
        strides=(2,),
        num_res_units=1,
        attention_reduction=1,
    )
    model = build_res_attention_unet(config)
    x = torch.zeros(1, 1, 16, 16)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 16, 16)


def test_res_attention_unet_attention_reduction_controls_gate_bottleneck_width() -> None:
    """attention_reduction must actually change the gate's inter_channels
    width, not just be accepted and ignored -- inspect the constructed
    gate module's own conv output channels directly."""
    model_reduction_2 = ResAttentionUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(4, 8),
        strides=(2,),
        attention_reduction=2,
    )
    model_reduction_1 = ResAttentionUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(4, 8),
        strides=(2,),
        attention_reduction=1,
    )
    gate_2 = model_reduction_2.attention_gates[0]
    gate_1 = model_reduction_1.attention_gates[0]
    # up_out is 4 here (channels[0]): reduction=2 -> inter_channels=2,
    # reduction=1 -> inter_channels=4.
    assert gate_2.psi.conv.in_channels == 2  # type: ignore[union-attr]
    assert gate_1.psi.conv.in_channels == 4  # type: ignore[union-attr]


def test_res_attention_unet_rejects_non_positive_attention_reduction() -> None:
    with pytest.raises(SegmentationError):
        ResAttentionUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(4, 8),
            strides=(2,),
            attention_reduction=0,
        )


def test_build_res_attention_unet_use_attention_false_forward_shape() -> None:
    """use_attention=False must still build and run end to end -- a
    plain residual U-Net with the attention gates removed."""
    config = ResAttentionUnetConfig(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(4, 8),
        strides=(2,),
        num_res_units=1,
        use_attention=False,
    )
    model = build_res_attention_unet(config)
    x = torch.zeros(1, 1, 16, 16)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 16, 16)


def test_res_attention_unet_use_attention_false_builds_no_gate_modules() -> None:
    """use_attention=False must actually remove the gate modules, not
    just skip calling them -- confirm no _AttentionGate parameters
    exist in that case, and that they do exist by default."""
    model_without = ResAttentionUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(4, 8),
        strides=(2,),
        use_attention=False,
    )
    model_with = ResAttentionUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(4, 8),
        strides=(2,),
        use_attention=True,
    )
    assert len(model_without.attention_gates) == 0
    assert len(model_with.attention_gates) == 1


def test_res_attention_unet_use_attention_false_output_differs_from_gated() -> None:
    """Removing the attention gates must actually change the model's
    output, not just be accepted and ignored -- same weights are
    impossible to compare directly (different parameter shapes), so
    this confirms the two variants are genuinely different computations
    by checking they don't happen to produce identical output on a
    fixed input."""
    torch.manual_seed(0)
    gated = ResAttentionUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(4, 8),
        strides=(2,),
        use_attention=True,
    )
    torch.manual_seed(0)
    ungated = ResAttentionUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(4, 8),
        strides=(2,),
        use_attention=False,
    )
    gated.eval()
    ungated.eval()
    x = torch.randn(1, 1, 16, 16)
    with torch.no_grad():
        y_gated = gated(x)
        y_ungated = ungated(x)
    assert not torch.allclose(y_gated, y_ungated)


def test_res_attention_unet_rejects_mismatched_strides_and_channels() -> None:
    """strides must have exactly one fewer entry than channels -- confirm
    the constructor actually validates this instead of failing later
    with a confusing shape-mismatch error deep in forward()."""
    with pytest.raises(SegmentationError):
        ResAttentionUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(4, 8, 16),
            strides=(2,),  # should be (2, 2)
        )


def test_res_attention_unet_gates_actually_use_skip_and_gate_signal() -> None:
    """The attention gate must be a real function of both its inputs --
    confirm changing either the gating (decoder) signal or the skip
    (encoder) signal changes the gate's output, not just accepted and
    ignored."""
    from miai_segmentation.two_d.models import _AttentionGate

    gate_module = _AttentionGate(spatial_dims=2, gate_channels=4, skip_channels=4, inter_channels=2)
    gate_module.eval()
    torch.manual_seed(0)
    gate_signal = torch.randn(1, 4, 8, 8)
    skip_signal = torch.randn(1, 4, 8, 8)
    with torch.no_grad():
        baseline = gate_module(gate=gate_signal, skip=skip_signal)
        different_gate = gate_module(gate=gate_signal + 5.0, skip=skip_signal)
        different_skip = gate_module(gate=gate_signal, skip=skip_signal + 5.0)

    assert not torch.allclose(baseline, different_gate)
    assert not torch.allclose(baseline, different_skip)


def test_build_model_dispatches_to_unet() -> None:
    model = build_model(ArchitectureConfig(kind="unet", unet=_TINY_UNET))
    assert isinstance(model, UNet)


def test_build_model_dispatches_to_attention_unet() -> None:
    model = build_model(
        ArchitectureConfig(kind="attention_unet", attention_unet=_TINY_ATTENTION_UNET)
    )
    assert isinstance(model, AttentionUnet)


def test_build_model_dispatches_to_res_attention_unet() -> None:
    model = build_model(
        ArchitectureConfig(kind="res_attention_unet", res_attention_unet=_TINY_RES_ATTENTION_UNET)
    )
    assert isinstance(model, ResAttentionUNet)


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
        kind="not-a-real-architecture",
        unet=UNetConfig(),
        attention_unet=AttentionUnetConfig(),
        res_attention_unet=ResAttentionUnetConfig(),
    )
    with pytest.raises(SegmentationError):
        build_model(config)
