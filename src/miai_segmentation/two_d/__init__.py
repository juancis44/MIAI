"""MIAI Segmentation -- 2D modality: per-slice architectures.

Two representative 2D architectures, both dispatched through
:class:`ArchitectureConfig`/:func:`build_model` so callers don't need to
know which one a given experiment picked: :class:`UNetConfig`/
:func:`build_unet` (residual-unit encoder/decoder, ``spatial_dims=2``)
and :class:`AttentionUnetConfig`/:func:`build_attention_unet`
(attention-gated skip connections, Oktay et al. 2018). Training
(:mod:`miai_segmentation.two_d.train`, re-exported from
:mod:`miai_segmentation.three_d.train` -- the loop is dimension-agnostic)
and inference (:mod:`miai_segmentation.two_d.infer`, a 2D-window variant
of :mod:`miai_segmentation.three_d.infer`) work with either architecture.
"""

from miai_segmentation.two_d.infer import InferenceConfig, run_case_inference, run_inference
from miai_segmentation.two_d.models import (
    ArchitectureConfig,
    AttentionUnetConfig,
    UNetConfig,
    build_attention_unet,
    build_model,
    build_unet,
)
from miai_segmentation.two_d.train import TrainingConfig, train_model

__all__ = [
    "ArchitectureConfig",
    "build_model",
    "UNetConfig",
    "build_unet",
    "AttentionUnetConfig",
    "build_attention_unet",
    "TrainingConfig",
    "train_model",
    "InferenceConfig",
    "run_inference",
    "run_case_inference",
]
