"""Shared pytest fixtures and synthetic-data helpers, used across
miai_dicom, miai_pipeline, miai_datasets, and miai_segmentation tests.
"""

from __future__ import annotations

from pathlib import Path

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


def make_dicom_series(
    directory,
    *,
    num_slices: int = 4,
    rows: int = 16,
    columns: int = 16,
    pixel_spacing: tuple[float, float] = (1.0, 1.0),
    slice_thickness: float = 2.0,
    modality: str = "CT",
    series_instance_uid: str | None = None,
):
    """Write a synthetic, multi-slice DICOM series (with real pixel data)
    to ``directory`` and return the list of file paths written.

    Used by miai_pipeline tests, which need a series SimpleITK's
    ``ImageSeriesReader`` can actually load (geometry + pixel data), not
    just the tag-only datasets ``make_dicom_dataset`` provides.
    """
    import numpy as np
    from pydicom.uid import generate_uid

    series_instance_uid = series_instance_uid or generate_uid()
    study_instance_uid = generate_uid()
    paths = []

    for i in range(num_slices):
        dataset = make_dicom_dataset(
            series_instance_uid=series_instance_uid,
            study_instance_uid=study_instance_uid,
            modality=modality,
            instance_number=i + 1,
            rows=rows,
            columns=columns,
        )
        dataset.PixelSpacing = list(pixel_spacing)
        dataset.SliceThickness = slice_thickness
        dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        dataset.ImagePositionPatient = [0.0, 0.0, float(i) * slice_thickness]
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 1
        dataset.RescaleIntercept = 0
        dataset.RescaleSlope = 1

        pixel_array = np.ones((rows, columns), dtype=np.int16) * (i + 1) * 100
        dataset.PixelData = pixel_array.tobytes()

        path = directory / f"slice_{i:03d}.dcm"
        dataset.save_as(str(path))
        paths.append(path)

    return paths


def make_synthetic_volume_pair(
    directory,
    *,
    name: str = "case0",
    size: tuple[int, int, int] = (16, 16, 16),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    """Write a synthetic image + binary label NIfTI pair to ``directory``.

    Used by miai_datasets / miai_segmentation tests, which need real
    files on disk for MONAI's ``LoadImaged`` transform to read (not just
    in-memory arrays). The label is a solid cube in the center of an
    otherwise-empty volume; the image is that same cube (scaled) plus
    Gaussian noise, so a model has a real, easy-to-fit signal to learn
    from in tests that train for a couple of epochs.

    Args:
        directory: Directory the two files are written under (created
            if missing).
        name: Case identifier used in the output filenames.
        size: Volume shape as ``(depth, height, width)``, matching
            SimpleITK's array convention.
        spacing: Voxel spacing in millimeters.

    Returns:
        ``(image_path, label_path)``.
    """
    import numpy as np
    import SimpleITK as sitk

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    depth, height, width = size
    label_arr = np.zeros((depth, height, width), dtype=np.uint8)
    d0, d1 = depth // 4, depth * 3 // 4
    h0, h1 = height // 4, height * 3 // 4
    w0, w1 = width // 4, width * 3 // 4
    label_arr[d0:d1, h0:h1, w0:w1] = 1

    image_arr = label_arr.astype(np.float32) * 2.0 + rng.normal(0, 0.1, size=size).astype(
        np.float32
    )

    image = sitk.GetImageFromArray(image_arr)
    image.SetSpacing(spacing)
    label = sitk.GetImageFromArray(label_arr)
    label.CopyInformation(image)

    image_path = directory / f"{name}_image.nii.gz"
    label_path = directory / f"{name}_label.nii.gz"
    sitk.WriteImage(image, str(image_path))
    sitk.WriteImage(label, str(label_path))
    return image_path, label_path
