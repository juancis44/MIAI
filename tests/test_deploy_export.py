"""Tests for export_model (TorchScript / ONNX export)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from miai_deploy.exceptions import DeployError
from miai_deploy.export import ExportConfig, export_model
from miai_segmentation.three_d.models import UNetConfig, build_unet

_UNET_CONFIG = UNetConfig(channels=(4, 8), strides=(2,), num_res_units=1)
_INPUT_SHAPE = (1, 1, 8, 8, 8)


def _make_checkpoint(tmp_path: Path) -> Path:
    checkpoint_path = tmp_path / "model.pt"
    torch.save(build_unet(_UNET_CONFIG).state_dict(), checkpoint_path)
    return checkpoint_path


@pytest.mark.slow
def test_export_torchscript_writes_loadable_module(tmp_path: Path) -> None:
    checkpoint_path = _make_checkpoint(tmp_path)
    config = ExportConfig(format="torchscript", example_input_shape=_INPUT_SHAPE)

    out_path = export_model(
        build_unet(_UNET_CONFIG), str(checkpoint_path), config, str(tmp_path / "model.pt")
    )

    assert out_path.exists()
    loaded = torch.jit.load(str(out_path))
    output = loaded(torch.zeros(_INPUT_SHAPE))
    assert output.shape[0] == _INPUT_SHAPE[0]


@pytest.mark.slow
def test_export_onnx_writes_file(tmp_path: Path) -> None:
    checkpoint_path = _make_checkpoint(tmp_path)
    config = ExportConfig(format="onnx", example_input_shape=_INPUT_SHAPE)

    out_path = export_model(
        build_unet(_UNET_CONFIG), str(checkpoint_path), config, str(tmp_path / "model.onnx")
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_unknown_export_format_raises(tmp_path: Path) -> None:
    checkpoint_path = _make_checkpoint(tmp_path)
    config = ExportConfig(example_input_shape=_INPUT_SHAPE).model_copy(
        update={"format": "not_a_real_format"}
    )

    with pytest.raises(DeployError):
        export_model(
            build_unet(_UNET_CONFIG), str(checkpoint_path), config, str(tmp_path / "model.out")
        )
