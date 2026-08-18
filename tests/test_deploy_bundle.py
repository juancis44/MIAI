"""Tests for write_bundle (export + reproducibility metadata)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from miai_deploy.bundle import BundleMetadata, write_bundle
from miai_deploy.export import ExportConfig
from miai_segmentation.three_d.models import UNetConfig, build_unet

_UNET_CONFIG = UNetConfig(channels=(4, 8), strides=(2,), num_res_units=1)
_INPUT_SHAPE = (1, 1, 8, 8, 8)


@pytest.mark.slow
def test_write_bundle_creates_model_and_metadata(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.pt"
    torch.save(build_unet(_UNET_CONFIG).state_dict(), checkpoint_path)

    metadata = BundleMetadata(
        name="test-unet", version="0.1.0", description="A tiny test bundle.", extra={"foo": "bar"}
    )

    bundle_dir = write_bundle(
        build_unet(_UNET_CONFIG),
        str(checkpoint_path),
        ExportConfig(format="torchscript", example_input_shape=_INPUT_SHAPE),
        metadata,
        str(tmp_path / "bundle"),
    )

    assert (bundle_dir / "model.pt").exists()
    assert (bundle_dir / "metadata.yaml").exists()

    reloaded = BundleMetadata.from_yaml(bundle_dir / "metadata.yaml")
    assert reloaded == metadata
