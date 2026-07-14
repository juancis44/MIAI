"""Integration test for the concrete EvaluationStage."""

from pathlib import Path

import pytest

from conftest import make_synthetic_volume_pair
from miai_evaluation.metrics import MetricsConfig
from miai_pipeline.context import PipelineContext
from miai_pipeline.exceptions import StageError
from miai_pipeline.stages.evaluation import EvaluationStage, EvaluationStageConfig


def test_evaluation_stage_scores_predictions_against_manifest_labels(tmp_path: Path) -> None:
    _, label_path = make_synthetic_volume_pair(tmp_path, name="case0")

    ctx = PipelineContext()
    ctx.set("prediction_paths", [str(label_path)])
    ctx.set(
        "manifest",
        {"train": [], "val": [], "test": [{"image": str(label_path), "label": str(label_path)}]},
    )

    stage = EvaluationStage(
        EvaluationStageConfig(metrics=MetricsConfig(), report_path=str(tmp_path / "report.json"))
    )
    result = stage.run(ctx)

    metrics = result.require("metrics")
    assert metrics["mean"]["dice"] == pytest.approx(1.0)
    assert (tmp_path / "report.json").exists()


def test_evaluation_stage_requires_labels_in_manifest(tmp_path: Path) -> None:
    _, label_path = make_synthetic_volume_pair(tmp_path, name="case0")

    ctx = PipelineContext()
    ctx.set("prediction_paths", [str(label_path)])
    ctx.set("manifest", {"train": [], "val": [], "test": [str(label_path)]})

    stage = EvaluationStage(EvaluationStageConfig())

    with pytest.raises(StageError):
        stage.run(ctx)
