"""Tests for the remaining Phase 4 placeholder stage (evaluation) and the
stage registry's naming, now that training/inference have concrete,
MONAI-backed implementations (see test_pipeline_training_stage.py and
test_pipeline_inference_stage.py).
"""

import pytest

from miai_pipeline.context import PipelineContext
from miai_pipeline.stages.evaluation import EvaluationStage
from miai_pipeline.stages.inference import InferenceStage
from miai_pipeline.stages.training import TrainingStage


def test_evaluation_placeholder_raises_not_implemented() -> None:
    stage = EvaluationStage()
    with pytest.raises(NotImplementedError, match="miai-evaluation|roadmap"):
        stage.run(PipelineContext())


@pytest.mark.parametrize(
    ("stage_cls", "expected_name"),
    [(TrainingStage, "training"), (InferenceStage, "inference"), (EvaluationStage, "evaluation")],
)
def test_stage_registered_under_expected_name(stage_cls: type, expected_name: str) -> None:
    from miai_pipeline.stages import STAGE_REGISTRY

    assert STAGE_REGISTRY[expected_name] is stage_cls
