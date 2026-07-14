"""Tests for miai_transforms.sitk_transforms.LoadImageSitkd."""

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from conftest import make_synthetic_volume_pair
from miai_transforms.sitk_transforms import LoadImageSitkd


def test_load_image_sitkd_matches_simpleitk_array(tmp_path: Path) -> None:
    image_path, label_path = make_synthetic_volume_pair(tmp_path, size=(8, 10, 12))

    loader = LoadImageSitkd(keys=["image", "label"])
    result = loader({"image": str(image_path), "label": str(label_path)})

    reference = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path))).astype(np.float32)
    assert result["image"].shape == (1, *reference.shape)
    np.testing.assert_array_equal(result["image"][0], reference)
    assert result["image"].dtype == np.float32


def test_load_image_sitkd_meta_dict_has_expected_keys(tmp_path: Path) -> None:
    image_path, _ = make_synthetic_volume_pair(tmp_path)
    loader = LoadImageSitkd(keys=["image"])
    result = loader({"image": str(image_path)})

    meta = result["image_meta_dict"]
    assert meta["filename_or_obj"] == str(image_path)
    assert len(meta["spacing"]) == 3
    assert len(meta["origin"]) == 3
    assert len(meta["direction"]) == 9


def test_load_image_sitkd_only_touches_configured_keys(tmp_path: Path) -> None:
    image_path, label_path = make_synthetic_volume_pair(tmp_path)
    loader = LoadImageSitkd(keys=["image"])
    result = loader({"image": str(image_path), "label": str(label_path)})

    assert isinstance(result["image"], np.ndarray)
    assert result["label"] == str(label_path)
