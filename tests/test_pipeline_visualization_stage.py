"""Integration test for the concrete VisualizationStage."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import make_offset_cube_volume
from miai_pipeline.context import PipelineContext
from miai_pipeline.exceptions import StageError
from miai_pipeline.stages.visualization import VisualizationStage, VisualizationStageConfig
from miai_visualization.slices import PlotMontageConfig


def test_visualization_stage_writes_qc_montage(tmp_path: Path) -> None:
    image_path = make_offset_cube_volume(tmp_path / "data", name="case0", size=(8, 8, 8))
    ctx = PipelineContext()
    ctx.set("preprocessed_paths", [image_path])

    stage = VisualizationStage(
        VisualizationStageConfig(
            output_dir=str(tmp_path / "qc"),
            montage=PlotMontageConfig(num_slices=4),
        )
    )

    result = stage.run(ctx)

    qc_paths = result.require("qc_visualization_paths")
    assert len(qc_paths) == 1
    assert Path(qc_paths[0]).exists()


def test_visualization_stage_empty_context_key_raises(tmp_path: Path) -> None:
    ctx = PipelineContext()
    ctx.set("preprocessed_paths", [])

    stage = VisualizationStage(VisualizationStageConfig(output_dir=str(tmp_path / "qc")))

    with pytest.raises(StageError):
        stage.run(ctx)
