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

As of Phase 4, training
(:class:`~miai_pipeline.stages.training.TrainingStage`) and inference
(:class:`~miai_pipeline.stages.inference.InferenceStage`) are concrete,
MONAI-backed implementations (see :mod:`miai_transforms`,
:mod:`miai_datasets`, and :mod:`miai_segmentation`). Evaluation
(:class:`~miai_pipeline.stages.evaluation.EvaluationStage`) remains an
interface only; its concrete implementation lands alongside
``miai-evaluation`` in a later phase. See docs/roadmap.md.
"""

from miai_pipeline.config import PipelineConfig, StageConfig
from miai_pipeline.context import PipelineContext
from miai_pipeline.exceptions import PipelineError, StageError, UnknownStageError
from miai_pipeline.pipeline import Pipeline
from miai_pipeline.stage import PipelineStage

__version__ = "0.4.0"

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
