"""Tests for miai_dicom.io."""

from pathlib import Path

import pydicom
import pytest

from conftest import make_dicom_dataset
from miai_core.exceptions import NotFoundError
from miai_dicom.exceptions import InvalidDicomFileError
from miai_dicom.io import is_dicom_file, read_dicom, write_dicom


def test_write_then_read_dicom_roundtrip(tmp_path: Path) -> None:
    dataset = make_dicom_dataset(patient_id="PAT123", modality="MR")
    path = tmp_path / "image.dcm"

    write_dicom(dataset, path)
    reloaded = read_dicom(path)

    assert reloaded.PatientID == "PAT123"
    assert reloaded.Modality == "MR"


def test_write_dicom_creates_parent_directories(tmp_path: Path) -> None:
    dataset = make_dicom_dataset()
    path = tmp_path / "nested" / "dir" / "image.dcm"

    write_dicom(dataset, path)

    assert path.exists()


def test_write_dicom_without_file_meta_raises_invalid_dicom_error(tmp_path: Path) -> None:
    # A bare Dataset has no file_meta (no Transfer Syntax UID), so
    # pydicom can't determine how to encode it -- save_as raises
    # ValueError, which write_dicom wraps as InvalidDicomFileError.
    dataset = pydicom.Dataset()
    dataset.PatientID = "PAT001"

    with pytest.raises(InvalidDicomFileError):
        write_dicom(dataset, tmp_path / "bad.dcm")


def test_read_dicom_missing_file_raises_not_found(tmp_path: Path) -> None:
    with pytest.raises(NotFoundError):
        read_dicom(tmp_path / "missing.dcm")


def test_read_dicom_invalid_file_raises_invalid_dicom_error(tmp_path: Path) -> None:
    path = tmp_path / "not_dicom.dcm"
    path.write_text("this is not a dicom file", encoding="utf-8")

    with pytest.raises(InvalidDicomFileError):
        read_dicom(path)


def test_is_dicom_file_true_for_valid_dicom(tmp_path: Path) -> None:
    dataset = make_dicom_dataset()
    path = tmp_path / "image.dcm"
    write_dicom(dataset, path)

    assert is_dicom_file(path) is True


def test_is_dicom_file_false_for_non_dicom(tmp_path: Path) -> None:
    path = tmp_path / "not_dicom.dcm"
    path.write_text("plain text", encoding="utf-8")

    assert is_dicom_file(path) is False


def test_is_dicom_file_false_for_missing_file(tmp_path: Path) -> None:
    assert is_dicom_file(tmp_path / "missing.dcm") is False
