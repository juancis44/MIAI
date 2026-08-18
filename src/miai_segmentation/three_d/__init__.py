"""MIAI Segmentation -- 3D modality: full-volume architectures.

Two representative 3D architectures, both dispatched through
:class:`ArchitectureConfig`/:func:`build_model` so callers don't need to
know which one a given experiment picked: :class:`UNetConfig`/
:func:`build_unet` (residual-unit encoder/decoder, MIAI's original
reference model) and :class:`SegResNetConfig`/:func:`build_segresnet`
(Myronenko 2018's residual-block encoder/decoder). Training
(:mod:`miai_segmentation.three_d.train`) and inference
(:mod:`miai_segmentation.three_d.infer`) are dimension-agnostic and work
with either.
"""

from miai_segmentation.three_d.infer import InferenceConfig, run_inference
from miai_segmentation.three_d.models import (
    ArchitectureConfig,
    SegResNetConfig,
    UNetConfig,
    build_model,
    build_segresnet,
    build_unet,
)
from miai_segmentation.three_d.train import TrainingConfig, train_model

__all__ = [
    "ArchitectureConfig",
    "build_model",
    "UNetConfig",
    "build_unet",
    "SegResNetConfig",
    "build_segresnet",
    "TrainingConfig",
    "train_model",
    "InferenceConfig",
    "run_inference",
]
