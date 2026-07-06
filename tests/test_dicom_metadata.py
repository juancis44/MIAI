"""Tests for miai_dicom.metadata."""

from conftest import make_dicom_dataset
from miai_dicom.metadata import extract_metadata


def test_extract_metadata_returns_core_fields() -> None:
    dataset = make_dicom_dataset(patient_id="PAT001", modality="CT", rows=128, columns=128)

    metadata = extract_metadata(dataset)

    assert metadata["patient_id"] == "PAT001"
    assert metadata["modality"] == "CT"
    assert metadata["rows"] == 128
    assert metadata["columns"] == 128
    assert metadata["series_instance_uid"] == dataset.SeriesInstanceUID


def test_extract_metadata_excludes_patient_name_by_default() -> None:
    dataset = make_dicom_dataset(patient_name="Doe^Jane")

    metadata = extract_metadata(dataset)

    assert "patient_name" not in metadata


def test_extract_metadata_includes_patient_name_when_requested() -> None:
    dataset = make_dicom_dataset(patient_name="Doe^Jane")

    metadata = extract_metadata(dataset, include_patient_name=True)

    assert metadata["patient_name"] == "Doe^Jane"


def test_extract_metadata_missing_tag_is_none() -> None:
    dataset = make_dicom_dataset()

    metadata = extract_metadata(dataset)

    assert metadata["slice_thickness"] is None


def test_extract_metadata_values_are_json_serializable() -> None:
    import json

    dataset = make_dicom_dataset()
    metadata = extract_metadata(dataset)

    json.dumps(metadata)  # should not raise
