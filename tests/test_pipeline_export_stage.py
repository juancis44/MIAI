"""Integration test for the concrete ExportStage."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from miai_core.exceptions import NotFoundError
from miai_deploy.bundle import BundleMetadata
from miai_deploy.export import ExportConfig
from miai_pipeline.context import PipelineContext
from miai_pipeline.stages.export import ExportStage, ExportStageConfig
from miai_segmentation.three_d.models import ArchitectureConfig, UNetConfig, build_unet

_UNET_CONFIG = UNetConfig(channels=(4, 8), strides=(2,), num_res_units=1)
_ARCHITECTURE_CONFIG = ArchitectureConfig(kind="unet", unet=_UNET_CONFIG)


@pytest.mark.slow
def test_export_stage_writes_bundle_using_context_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.pt"
    torch.save(build_unet(_UNET_CONFIG).state_dict(), checkpoint_path)

    ctx = PipelineContext()
    ctx.set("model_checkpoint_path", str(checkpoint_path))

    stage = ExportStage(
        ExportStageConfig(
            output_dir=str(tmp_path / "bundle"),
            architecture=_ARCHITECTURE_CONFIG,
            export=ExportConfig(format="torchscript", example_input_shape=(1, 1, 8, 8, 8)),
            metadata=BundleMetadata(name="test-unet", version="0.1.0"),
        )
    )

    result = stage.run(ctx)

    bundle_path = result.require("deploy_bundle_path")
    assert (Path(bundle_path) / "model.pt").exists()
    assert (Path(bundle_path) / "metadata.yaml").exists()


def test_export_stage_missing_checkpoint_raises(tmp_path: Path) -> None:
    ctx = PipelineContext()

    stage = ExportStage(
        ExportStageConfig(
            output_dir=str(tmp_path / "bundle"),
            metadata=BundleMetadata(name="test-unet", version="0.1.0"),
        )
    )

    with pytest.raises(NotFoundError):
        stage.run(ctx)
