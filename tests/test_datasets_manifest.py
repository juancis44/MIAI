"""Tests for miai_datasets.manifest."""

import pytest

from miai_datasets.exceptions import DatasetBuildError
from miai_datasets.manifest import manifest_split_to_data_dicts


def test_normalizes_plain_string_entries() -> None:
    result = manifest_split_to_data_dicts(["a.nii.gz", "b.nii.gz"])
    assert result == [{"image": "a.nii.gz"}, {"image": "b.nii.gz"}]


def test_normalizes_image_label_dict_entries() -> None:
    result = manifest_split_to_data_dicts([{"image": "a.nii.gz", "label": "a_seg.nii.gz"}])
    assert result == [{"image": "a.nii.gz", "label": "a_seg.nii.gz"}]


def test_dict_entry_missing_image_key_raises() -> None:
    with pytest.raises(DatasetBuildError, match="image"):
        manifest_split_to_data_dicts([{"label": "a_seg.nii.gz"}])


def test_invalid_entry_type_raises() -> None:
    with pytest.raises(DatasetBuildError, match="path string or a mapping"):
        manifest_split_to_data_dicts([123])


def test_empty_split_returns_empty_list() -> None:
    assert manifest_split_to_data_dicts([]) == []
