"""Tests for miai_transforms.slice_transforms (ExtractSliced/ExtractSliceStackd)."""

from collections.abc import Hashable
from typing import Any

import numpy as np
import pytest

from miai_transforms.exceptions import TransformError
from miai_transforms.slice_transforms import ExtractSliced, ExtractSliceStackd


def _volume(depth: int = 5, height: int = 4, width: int = 3) -> np.ndarray:
    # (C, D, H, W), each depth slice filled with its own index so slicing
    # can be verified by value rather than just by shape.
    volume = np.zeros((1, depth, height, width), dtype=np.float32)
    for z in range(depth):
        volume[0, z, :, :] = z
    return volume


def test_extract_sliced_returns_expected_slice() -> None:
    transform = ExtractSliced(keys=["image"])
    data: dict[Hashable, Any] = {"image": _volume(), "slice_index": "2"}

    result = transform(data)

    assert result["image"].shape == (1, 4, 3)
    np.testing.assert_array_equal(result["image"], np.full((1, 4, 3), 2))


def test_extract_sliced_clamps_out_of_range_index() -> None:
    transform = ExtractSliced(keys=["image"])
    data: dict[Hashable, Any] = {"image": _volume(depth=5), "slice_index": "99"}

    result = transform(data)

    np.testing.assert_array_equal(result["image"], np.full((1, 4, 3), 4))


def test_extract_sliced_clamps_negative_index() -> None:
    transform = ExtractSliced(keys=["image"])
    data: dict[Hashable, Any] = {"image": _volume(depth=5), "slice_index": "-3"}

    result = transform(data)

    np.testing.assert_array_equal(result["image"], np.full((1, 4, 3), 0))


def test_extract_sliced_only_touches_configured_keys() -> None:
    transform = ExtractSliced(keys=["image"])
    label_volume = _volume()
    data: dict[Hashable, Any] = {"image": _volume(), "label": label_volume, "slice_index": "1"}

    result = transform(data)

    assert result["image"].shape == (1, 4, 3)
    assert result["label"] is label_volume


def test_extract_slice_stackd_stacks_adjacent_slices() -> None:
    transform = ExtractSliceStackd(keys=["image"], context_slices=3)
    data: dict[Hashable, Any] = {"image": _volume(depth=5), "slice_index": "2"}

    result = transform(data)

    assert result["image"].shape == (3, 4, 3)
    np.testing.assert_array_equal(result["image"][0], np.full((4, 3), 1))
    np.testing.assert_array_equal(result["image"][1], np.full((4, 3), 2))
    np.testing.assert_array_equal(result["image"][2], np.full((4, 3), 3))


def test_extract_slice_stackd_clamps_at_volume_start() -> None:
    transform = ExtractSliceStackd(keys=["image"], context_slices=3)
    data: dict[Hashable, Any] = {"image": _volume(depth=5), "slice_index": "0"}

    result = transform(data)

    # Center index 0, offsets -1/0/+1 clamp to 0/0/1.
    np.testing.assert_array_equal(result["image"][0], np.full((4, 3), 0))
    np.testing.assert_array_equal(result["image"][1], np.full((4, 3), 0))
    np.testing.assert_array_equal(result["image"][2], np.full((4, 3), 1))


def test_extract_slice_stackd_clamps_at_volume_end() -> None:
    transform = ExtractSliceStackd(keys=["image"], context_slices=3)
    data: dict[Hashable, Any] = {"image": _volume(depth=5), "slice_index": "4"}

    result = transform(data)

    np.testing.assert_array_equal(result["image"][0], np.full((4, 3), 3))
    np.testing.assert_array_equal(result["image"][1], np.full((4, 3), 4))
    np.testing.assert_array_equal(result["image"][2], np.full((4, 3), 4))


def test_extract_slice_stackd_rejects_even_context_slices() -> None:
    with pytest.raises(TransformError):
        ExtractSliceStackd(keys=["image"], context_slices=4)


def test_extract_slice_stackd_rejects_non_positive_context_slices() -> None:
    with pytest.raises(TransformError):
        ExtractSliceStackd(keys=["image"], context_slices=0)
