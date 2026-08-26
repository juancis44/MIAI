"""Tests for miai_registration.register."""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import SimpleITK as sitk

from miai_registration.exceptions import RegistrationError
from miai_registration.register import RegistrationConfig, register_images

_FAST_CONFIG = RegistrationConfig(
    transform_type="rigid",
    metric="mean_squares",
    number_of_iterations=150,
    sampling_percentage=1.0,
    shrink_factors=(1,),
    smoothing_sigmas=(0.0,),
)


def _cube_image(
    shift: tuple[int, int, int] = (0, 0, 0), size: int = 32
) -> tuple[sitk.Image, npt.NDArray[Any]]:
    arr = np.zeros((size, size, size), dtype=np.float32)
    lo, hi = size // 3, size // 3 * 2
    d0, d1 = lo + shift[0], hi + shift[0]
    h0, h1 = lo + shift[1], hi + shift[1]
    w0, w1 = lo + shift[2], hi + shift[2]
    arr[d0:d1, h0:h1, w0:w1] = 100.0
    return sitk.GetImageFromArray(arr), arr


def test_register_images_recovers_small_translation() -> None:
    fixed, fixed_arr = _cube_image(shift=(0, 0, 0))
    moving, _ = _cube_image(shift=(3, 0, 0))

    _, registered = register_images(fixed, moving, _FAST_CONFIG)

    registered_arr = sitk.GetArrayFromImage(registered)
    pred_fg = registered_arr > 50
    gt_fg = fixed_arr > 50
    dice = 2 * np.sum(pred_fg & gt_fg) / (np.sum(pred_fg) + np.sum(gt_fg))
    assert dice > 0.95


def test_register_images_output_matches_fixed_grid() -> None:
    fixed, _ = _cube_image(shift=(0, 0, 0))
    moving, _ = _cube_image(shift=(2, 0, 0))

    _, registered = register_images(fixed, moving, _FAST_CONFIG)

    assert registered.GetSize() == fixed.GetSize()
    assert registered.GetSpacing() == fixed.GetSpacing()


@pytest.mark.parametrize("transform_type", ["affine", "bspline"])
def test_register_images_runs_with_each_transform_type(transform_type: str) -> None:
    fixed, _ = _cube_image(shift=(0, 0, 0))
    moving, _ = _cube_image(shift=(2, 0, 0))

    config = _FAST_CONFIG.model_copy(
        update={"transform_type": transform_type, "number_of_iterations": 5}
    )
    transform, registered = register_images(fixed, moving, config)

    assert transform is not None
    assert registered.GetSize() == fixed.GetSize()


@pytest.mark.parametrize("metric", ["mattes_mutual_information", "correlation"])
def test_register_images_runs_with_each_metric(metric: str) -> None:
    fixed, _ = _cube_image(shift=(0, 0, 0))
    moving, _ = _cube_image(shift=(2, 0, 0))

    config = _FAST_CONFIG.model_copy(update={"metric": metric, "number_of_iterations": 5})
    transform, registered = register_images(fixed, moving, config)

    assert transform is not None
    assert registered.GetSize() == fixed.GetSize()


def test_register_images_unknown_transform_type_raises() -> None:
    fixed, _ = _cube_image()
    moving, _ = _cube_image(shift=(1, 0, 0))

    config = _FAST_CONFIG.model_copy(update={"transform_type": "not_a_real_transform"})
    with pytest.raises(RegistrationError):
        register_images(fixed, moving, config)


def test_register_images_unknown_metric_raises() -> None:
    fixed, _ = _cube_image()
    moving, _ = _cube_image(shift=(1, 0, 0))

    config = _FAST_CONFIG.model_copy(update={"metric": "not_a_real_metric"})
    with pytest.raises(RegistrationError):
        register_images(fixed, moving, config)
