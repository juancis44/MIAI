"""Shared pytest fixtures for miai_dicom tests."""

from __future__ import annotations

from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


def make_dicom_dataset(
    *,
    patient_id: str = "PAT001",
    patient_name: str = "Doe^Jane",
    series_instance_uid: str | None = None,
    study_instance_uid: str | None = None,
    sop_instance_uid: str | None = None,
    modality: str = "CT",
    instance_number: int | None = None,
    rows: int = 64,
    columns: int = 64,
) -> FileDataset:
    """Build a minimal, valid, in-memory DICOM dataset for tests.

    Includes just enough ``file_meta`` and dataset-level tags to be
    written to disk and read back with pydicom, without requiring real
    pixel data.
    """
    sop_instance_uid = sop_instance_uid or generate_uid()

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dataset = FileDataset(
        filename_or_obj=None,
        dataset=Dataset(),
        file_meta=file_meta,
        preamble=b"\x00" * 128,
    )
    dataset.PatientID = patient_id
    dataset.PatientName = patient_name
    dataset.StudyInstanceUID = study_instance_uid or generate_uid()
    dataset.SeriesInstanceUID = series_instance_uid or generate_uid()
    dataset.SOPInstanceUID = sop_instance_uid
    dataset.Modality = modality
    dataset.Rows = rows
    dataset.Columns = columns
    if instance_number is not None:
        dataset.InstanceNumber = instance_number

    return dataset
