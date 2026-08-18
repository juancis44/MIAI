"""Training loop for a MIAI 2.5D binary segmentation model.

Re-exports :mod:`miai_segmentation.three_d.train` unchanged, same
rationale as :mod:`miai_segmentation.two_d.train`: the loop only calls
``model(inputs)`` and scores the result against a label of matching
shape, regardless of what the input channels represent (stacked slices
vs. a single-channel image), so it needs no 2.5D-specific variant.
"""

from __future__ import annotations

from miai_segmentation.three_d.train import TrainingConfig, train_model

__all__ = [
    "TrainingConfig",
    "train_model",
]
