"""Tests for miai_dicom.anonymize."""

from conftest import make_dicom_dataset
from miai_dicom.anonymize import anonymize


def test_anonymize_removes_patient_name() -> None:
    dataset = make_dicom_dataset(patient_name="Doe^Jane")

    result = anonymize(dataset)

    assert "PatientName" not in result


def test_anonymize_replaces_patient_id() -> None:
    dataset = make_dicom_dataset(patient_id="REAL_ID_123")

    result = anonymize(dataset)

    assert result.PatientID == "ANONYMIZED"


def test_anonymize_regenerates_uids_by_default() -> None:
    dataset = make_dicom_dataset()
    original_study_uid = dataset.StudyInstanceUID
    original_series_uid = dataset.SeriesInstanceUID
    original_sop_uid = dataset.SOPInstanceUID

    result = anonymize(dataset)

    assert result.StudyInstanceUID != original_study_uid
    assert result.SeriesInstanceUID != original_series_uid
    assert result.SOPInstanceUID != original_sop_uid


def test_anonymize_can_keep_uids() -> None:
    dataset = make_dicom_dataset()
    original_study_uid = dataset.StudyInstanceUID

    result = anonymize(dataset, regenerate_uids=False)

    assert result.StudyInstanceUID == original_study_uid


def test_anonymize_does_not_mutate_original_by_default() -> None:
    dataset = make_dicom_dataset(patient_name="Doe^Jane")

    anonymize(dataset)

    assert dataset.PatientName == "Doe^Jane"


def test_anonymize_in_place_mutates_original() -> None:
    dataset = make_dicom_dataset(patient_name="Doe^Jane")

    result = anonymize(dataset, in_place=True)

    assert result is dataset
    assert "PatientName" not in dataset


def test_anonymize_sets_deidentification_flags() -> None:
    dataset = make_dicom_dataset()

    result = anonymize(dataset)

    assert result.PatientIdentityRemoved == "YES"
    assert result.DeidentificationMethod


def test_anonymize_custom_tags_to_remove() -> None:
    dataset = make_dicom_dataset(modality="CT")

    result = anonymize(dataset, tags_to_remove=["Modality"], tags_to_replace={})

    assert "Modality" not in result
    # PatientName untouched since it wasn't in the custom removal list
    assert result.PatientName == dataset.PatientName
