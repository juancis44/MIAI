"""Concrete pipeline stages, and the registry mapping a config file's
``type`` string to a stage class.
"""

from __future__ import annotations

from miai_pipeline.stage import PipelineStage
from miai_pipeline.stages.dataset import DatasetConfig, DatasetStage
from miai_pipeline.stages.denoising import DenoisingStage, DenoisingStageConfig
from miai_pipeline.stages.dicom_to_nifti import DicomToNiftiConfig, DicomToNiftiStage
from miai_pipeline.stages.diffusion_training import (
    DiffusionTrainingStage,
    DiffusionTrainingStageConfig,
)
from miai_pipeline.stages.evaluation import EvaluationStage, EvaluationStageConfig
from miai_pipeline.stages.inference import InferenceStage, InferenceStageConfig
from miai_pipeline.stages.preprocessing import PreprocessingConfig, PreprocessingStage
from miai_pipeline.stages.registration import RegistrationStage, RegistrationStageConfig
from miai_pipeline.stages.training import TrainingStage, TrainingStageConfig

#: Maps a :class:`~miai_pipeline.config.StageConfig.type` string to the
#: stage class that implements it. Used by
#: :func:`miai_pipeline.pipeline.build_pipeline` to construct a
#: :class:`~miai_pipeline.pipeline.Pipeline` from a
#: :class:`~miai_pipeline.config.PipelineConfig`.
STAGE_REGISTRY: dict[str, type[PipelineStage]] = {
    DicomToNiftiStage.name: DicomToNiftiStage,
    PreprocessingStage.name: PreprocessingStage,
    RegistrationStage.name: RegistrationStage,
    DatasetStage.name: DatasetStage,
    TrainingStage.name: TrainingStage,
    InferenceStage.name: InferenceStage,
    EvaluationStage.name: EvaluationStage,
    DiffusionTrainingStage.name: DiffusionTrainingStage,
    DenoisingStage.name: DenoisingStage,
}

__all__ = [
    "STAGE_REGISTRY",
    "DicomToNiftiStage",
    "DicomToNiftiConfig",
    "PreprocessingStage",
    "PreprocessingConfig",
    "RegistrationStage",
    "RegistrationStageConfig",
    "DatasetStage",
    "DatasetConfig",
    "TrainingStage",
    "TrainingStageConfig",
    "InferenceStage",
    "InferenceStageConfig",
    "EvaluationStage",
    "EvaluationStageConfig",
    "DiffusionTrainingStage",
    "DiffusionTrainingStageConfig",
    "DenoisingStage",
    "DenoisingStageConfig",
]
