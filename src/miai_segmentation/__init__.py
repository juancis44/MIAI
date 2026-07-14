"""MIAI Segmentation: reference binary 3D segmentation on top of MONAI.

Provides a MONAI :class:`~monai.networks.nets.UNet` builder
(:mod:`miai_segmentation.models`), a supervised training loop
(:mod:`miai_segmentation.train`), and sliding-window inference
(:mod:`miai_segmentation.infer`), each configured through a
:class:`~miai_core.config.MIAIBaseConfig` subclass so an experiment is
fully described by its YAML config. Used by
:class:`~miai_pipeline.stages.training.TrainingStage` and
:class:`~miai_pipeline.stages.inference.InferenceStage` to implement
the ``training`` / ``inference`` steps of the clinical workflow.
"""

from miai_segmentation.exceptions import SegmentationError
from miai_segmentation.infer import InferenceConfig, run_inference
from miai_segmentation.models import UNetConfig, build_unet
from miai_segmentation.train import TrainingConfig, train_model

__version__ = "0.1.0"

__all__ = [
    "build_unet",
    "UNetConfig",
    "train_model",
    "TrainingConfig",
    "run_inference",
    "InferenceConfig",
    "SegmentationError",
    "__version__",
]
