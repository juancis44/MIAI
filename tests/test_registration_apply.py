"""Tests for miai_registration.apply."""

import numpy as np
import SimpleITK as sitk

from miai_registration.apply import apply_transform
from miai_registration.register import RegistrationConfig, register_images

_FAST_CONFIG = RegistrationConfig(
    transform_type="rigid",
    metric="mean_squares",
    number_of_iterations=150,
    sampling_percentage=1.0,
    shrink_factors=(1,),
    smoothing_sigmas=(0.0,),
)


def test_apply_transform_propagates_registration_to_label() -> None:
    size = 32
    fixed_arr = np.zeros((size, size, size), dtype=np.float32)
    fixed_arr[10:20, 10:20, 10:20] = 100.0
    fixed = sitk.GetImageFromArray(fixed_arr)

    moving_arr = np.zeros((size, size, size), dtype=np.float32)
    moving_arr[13:23, 10:20, 10:20] = 100.0
    moving = sitk.GetImageFromArray(moving_arr)

    label_arr = (moving_arr > 50).astype(np.uint8)
    label = sitk.GetImageFromArray(label_arr)
    label.CopyInformation(moving)

    transform, _ = register_images(fixed, moving, _FAST_CONFIG)
    registered_label = apply_transform(label, fixed, transform, interpolator="nearest")

    registered_label_arr = sitk.GetArrayFromImage(registered_label)
    assert set(np.unique(registered_label_arr).tolist()).issubset({0, 1})

    gt_fg = fixed_arr > 50
    pred_fg = registered_label_arr == 1
    dice = 2 * np.sum(pred_fg & gt_fg) / (np.sum(pred_fg) + np.sum(gt_fg))
    assert dice > 0.95


def test_apply_transform_output_matches_reference_grid() -> None:
    size = 16
    fixed = sitk.Image(size, size, size, sitk.sitkFloat32)
    moving_arr = np.ones((size, size, size), dtype=np.uint8)
    moving = sitk.GetImageFromArray(moving_arr)

    identity = sitk.Transform(3, sitk.sitkIdentity)
    result = apply_transform(moving, fixed, identity, interpolator="nearest")

    assert result.GetSize() == fixed.GetSize()
