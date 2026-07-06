"""Tests for miai_dicom.validation."""

import pytest

from conftest import make_dicom_dataset
from miai_core.exceptions import ValidationError
from miai_dicom.validation import is_valid_dataset, validate_dataset


def test_validate_dataset_passes_for_complete_dataset() -> None:
    dataset = make_dicom_dataset()

    validate_dataset(dataset)  # should not raise


def test_validate_dataset_raises_for_missing_tags() -> None:
    dataset = make_dicom_dataset()
    del dataset.Rows

    with pytest.raises(ValidationError, match="Rows"):
        validate_dataset(dataset)


def test_validate_dataset_respects_custom_required_tags() -> None:
    dataset = make_dicom_dataset()

    validate_dataset(dataset, required_tags=("PatientID",))


def test_is_valid_dataset_true_for_complete_dataset() -> None:
    dataset = make_dicom_dataset()

    assert is_valid_dataset(dataset) is True


def test_is_valid_dataset_false_for_incomplete_dataset() -> None:
    dataset = make_dicom_dataset()
    del dataset.Modality

    assert is_valid_dataset(dataset) is False
