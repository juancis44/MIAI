"""Integration test for the concrete ReconstructionStage."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import make_offset_cube_volume
from miai_pipeline.context import PipelineContext
from miai_pipeline.exceptions import StageError
from miai_pipeline.stages.reconstruction import ReconstructionStage, ReconstructionStageConfig
from miai_reconstruction.kspace import UndersamplingConfig


def test_reconstruction_stage_writes_reconstructed_paths(tmp_path: Path) -> None:
    image_path = make_offset_cube_volume(tmp_path / "data", name="case0", size=(8, 8, 8))
    ctx = PipelineContext()
    ctx.set("preprocessed_paths", [image_path])

    stage = ReconstructionStage(
        ReconstructionStageConfig(
            output_dir=str(tmp_path / "reconstructed"),
            undersampling=UndersamplingConfig(acceleration=4.0),
        )
    )

    result = stage.run(ctx)

    reconstructed_paths = result.require("reconstructed_paths")
    assert len(reconstructed_paths) == 1
    assert Path(reconstructed_paths[0]).exists()


def test_reconstruction_stage_empty_context_key_raises(tmp_path: Path) -> None:
    ctx = PipelineContext()
    ctx.set("preprocessed_paths", [])

    stage = ReconstructionStage(
        ReconstructionStageConfig(output_dir=str(tmp_path / "reconstructed"))
    )

    with pytest.raises(StageError):
        stage.run(ctx)
