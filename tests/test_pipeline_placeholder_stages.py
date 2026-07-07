"""Tests for the Phase 4 placeholder stages (training/inference/evaluation)."""

import pytest

from miai_pipeline.context import PipelineContext
from miai_pipeline.stages.evaluation import EvaluationStage
from miai_pipeline.stages.inference import InferenceStage
from miai_pipeline.stages.training import TrainingStage


@pytest.mark.parametrize("stage_cls", [TrainingStage, InferenceStage, EvaluationStage])
def test_placeholder_stage_raises_not_implemented(stage_cls: type) -> None:
    stage = stage_cls()
    with pytest.raises(NotImplementedError, match="Phase 4|roadmap"):
        stage.run(PipelineContext())


@pytest.mark.parametrize(
    ("stage_cls", "expected_name"),
    [(TrainingStage, "training"), (InferenceStage, "inference"), (EvaluationStage, "evaluation")],
)
def test_placeholder_stage_registered_under_expected_name(
    stage_cls: type, expected_name: str
) -> None:
    from miai_pipeline.stages import STAGE_REGISTRY

    assert STAGE_REGISTRY[expected_name] is stage_cls
