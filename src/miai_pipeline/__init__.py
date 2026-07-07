"""MIAI Pipeline: config-driven orchestration of the clinical workflow.

    DICOM -> NIfTI -> Preprocessing -> Dataset -> Training -> Inference -> Evaluation

Each step is a :class:`~miai_pipeline.stage.PipelineStage` that reads
inputs from, and writes outputs to, a shared
:class:`~miai_pipeline.context.PipelineContext`. A
:class:`~miai_pipeline.pipeline.Pipeline` runs a list of stages in
order; :meth:`~miai_pipeline.pipeline.Pipeline.from_config` builds that
list from a YAML :class:`~miai_pipeline.config.PipelineConfig`, so an
experiment is defined by its config file rather than by editing code
(see docs/vision.md, "Reproducibility first").

Training, inference, and evaluation are defined here as interfaces only
(:class:`~miai_pipeline.stages.training.TrainingStage`,
:class:`~miai_pipeline.stages.inference.InferenceStage`,
:class:`~miai_pipeline.stages.evaluation.EvaluationStage`) — concrete
implementations land in Phase 4 once MONAI is integrated. See
docs/roadmap.md.
"""

from miai_pipeline.config import PipelineConfig, StageConfig
from miai_pipeline.context import PipelineContext
from miai_pipeline.exceptions import PipelineError, StageError, UnknownStageError
from miai_pipeline.pipeline import Pipeline
from miai_pipeline.stage import PipelineStage

__version__ = "0.1.0"

__all__ = [
    "Pipeline",
    "PipelineConfig",
    "StageConfig",
    "PipelineContext",
    "PipelineStage",
    "PipelineError",
    "StageError",
    "UnknownStageError",
    "__version__",
]
