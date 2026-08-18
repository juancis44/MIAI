"""Tests asserting the segmentation re-export modules point at the right place.

``miai_segmentation.two_d.train``/``.two_half_d.train`` re-export
``miai_segmentation.three_d.train`` (the training loop is
dimension-agnostic), and ``miai_segmentation.two_half_d.infer``
re-exports ``miai_segmentation.two_d.infer`` (both are 2D-window
inference). These tests exist so a future refactor that accidentally
breaks the re-export (e.g. shadowing the name with a new, divergent
definition) fails loudly instead of silently drifting.
"""

import miai_segmentation.three_d.train as three_d_train
import miai_segmentation.two_d.infer as two_d_infer
import miai_segmentation.two_d.train as two_d_train
import miai_segmentation.two_half_d.infer as two_half_d_infer
import miai_segmentation.two_half_d.train as two_half_d_train


def test_two_d_train_reexports_three_d_train() -> None:
    assert two_d_train.train_model is three_d_train.train_model
    assert two_d_train.TrainingConfig is three_d_train.TrainingConfig


def test_two_half_d_train_reexports_three_d_train() -> None:
    assert two_half_d_train.train_model is three_d_train.train_model
    assert two_half_d_train.TrainingConfig is three_d_train.TrainingConfig


def test_two_half_d_infer_reexports_two_d_infer() -> None:
    assert two_half_d_infer.run_inference is two_d_infer.run_inference
    assert two_half_d_infer.InferenceConfig is two_d_infer.InferenceConfig
