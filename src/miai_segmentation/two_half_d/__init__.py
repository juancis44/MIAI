"""MIAI Segmentation -- 2.5D modality: stacked-adjacent-slice architectures.

One architecture today, :class:`StackedUNetConfig`/
:func:`build_stacked_unet` (a 2D UNet whose ``in_channels`` is the
number of stacked adjacent slices, predicting the center slice's mask),
dispatched through :class:`ArchitectureConfig`/:func:`build_model` for
consistency with :mod:`miai_segmentation.three_d` and
:mod:`miai_segmentation.two_d`. Training
(:mod:`miai_segmentation.two_half_d.train`) and inference
(:mod:`miai_segmentation.two_half_d.infer`) are both re-exported from
elsewhere unchanged -- see each module's docstring for why nothing here
needs a 2.5D-specific implementation.
"""

from miai_segmentation.two_half_d.infer import InferenceConfig, run_inference
from miai_segmentation.two_half_d.models import (
    ArchitectureConfig,
    StackedUNetConfig,
    build_model,
    build_stacked_unet,
)
from miai_segmentation.two_half_d.train import TrainingConfig, train_model

__all__ = [
    "ArchitectureConfig",
    "build_model",
    "StackedUNetConfig",
    "build_stacked_unet",
    "TrainingConfig",
    "train_model",
    "InferenceConfig",
    "run_inference",
]
