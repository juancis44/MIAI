"""Tests for miai_datasets.slices.expand_to_slice_dicts."""

from pathlib import Path
from unittest.mock import patch

import pytest

from conftest import make_synthetic_volume_pair
from miai_datasets.exceptions import DatasetBuildError
from miai_datasets.slices import expand_to_slice_dicts


def test_expands_single_case_into_one_dict_per_slice(tmp_path: Path) -> None:
    image_path, label_path = make_synthetic_volume_pair(tmp_path, size=(5, 4, 3))

    slice_dicts, case_slice_counts = expand_to_slice_dicts(
        [{"image": str(image_path), "label": str(label_path)}]
    )

    assert case_slice_counts == [5]
    assert len(slice_dicts) == 5
    assert [d["slice_index"] for d in slice_dicts] == ["0", "1", "2", "3", "4"]
    for d in slice_dicts:
        assert d["image"] == str(image_path)
        assert d["label"] == str(label_path)


def test_expands_multiple_cases_in_case_major_order(tmp_path: Path) -> None:
    image0, _ = make_synthetic_volume_pair(tmp_path, name="case0", size=(3, 4, 3))
    image1, _ = make_synthetic_volume_pair(tmp_path, name="case1", size=(2, 4, 3))

    slice_dicts, case_slice_counts = expand_to_slice_dicts(
        [{"image": str(image0)}, {"image": str(image1)}]
    )

    assert case_slice_counts == [3, 2]
    assert [d["image"] for d in slice_dicts] == [str(image0)] * 3 + [str(image1)] * 2
    assert [d["slice_index"] for d in slice_dicts] == ["0", "1", "2", "0", "1"]


def test_empty_data_dicts_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetBuildError, match="empty"):
        expand_to_slice_dicts([])


def test_missing_image_key_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetBuildError, match="image"):
        expand_to_slice_dicts([{"label": "a_seg.nii.gz"}])


def test_zero_depth_volume_raises(tmp_path: Path) -> None:
    image_path, _ = make_synthetic_volume_pair(tmp_path, size=(3, 4, 3))

    # SimpleITK refuses to write a genuinely zero-depth NIfTI file to
    # disk, so the only way to exercise this guard is to make
    # _read_depth (which normally reads the real header) report 0.
    with (
        patch("miai_datasets.slices._read_depth", return_value=0),
        pytest.raises(DatasetBuildError, match="zero slices"),
    ):
        expand_to_slice_dicts([{"image": str(image_path)}])


def test_non_3d_volume_raises(tmp_path: Path) -> None:
    import numpy as np
    import SimpleITK as sitk

    path = tmp_path / "flat.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((4, 3), dtype=np.float32)), str(path))

    with pytest.raises(DatasetBuildError, match="3D"):
        expand_to_slice_dicts([{"image": str(path)}])
