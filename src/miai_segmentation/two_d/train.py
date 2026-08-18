"""Training loop for a MIAI 2D binary segmentation model.

Re-exports :mod:`miai_segmentation.three_d.train` unchanged.
:func:`~miai_segmentation.three_d.train.train_model` only calls
``model(inputs)`` and scores the result against a label with the same
shape -- nothing about it assumes ``spatial_dims=3`` -- so a second,
near-identical implementation here would just be duplicated
maintenance. If 2D training ever needs modality-specific behavior (e.g.
per-slice class-imbalance handling), give it its own function then;
until it does, this module is the modality's public training entry
point per `docs/api_design.md`, even though the implementation lives
elsewhere.
"""

from __future__ import annotations

from miai_segmentation.three_d.train import TrainingConfig, train_model

__all__ = [
    "TrainingConfig",
    "train_model",
]
