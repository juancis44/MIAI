"""Stage registry naming checks.

Historically this file also tested the Phase 4 placeholder stages
(training/inference/evaluation raising NotImplementedError). As of the
miai-evaluation package, all three have concrete implementations --
see test_pipeline_training_stage.py, test_pipeline_inference_stage.py,
and test_pipeline_evaluation_stage.py -- so only the registry-naming
check remains here.
"""

import pytest

from miai_pipeline.stages.evaluation import EvaluationStage
from miai_pipeline.stages.inference import InferenceStage
from miai_pipeline.stages.training import TrainingStage


@pytest.mark.parametrize(
    ("stage_cls", "expected_name"),
    [(TrainingStage, "training"), (InferenceStage, "inference"), (EvaluationStage, "evaluation")],
)
def test_stage_registered_under_expected_name(stage_cls: type, expected_name: str) -> None:
    from miai_pipeline.stages import STAGE_REGISTRY

    assert STAGE_REGISTRY[expected_name] is stage_cls
