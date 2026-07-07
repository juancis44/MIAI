"""Tests for miai_pipeline.stages.dicom_to_nifti."""

from pathlib import Path

from conftest import make_dicom_series
from miai_pipeline.context import PipelineContext
from miai_pipeline.stages.dicom_to_nifti import DicomToNiftiConfig, DicomToNiftiStage


def test_dicom_to_nifti_converts_series(tmp_path: Path) -> None:
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()
    make_dicom_series(dicom_dir, num_slices=4, rows=16, columns=16)

    output_dir = tmp_path / "nifti"
    stage = DicomToNiftiStage(DicomToNiftiConfig(output_dir=str(output_dir)))

    ctx = PipelineContext()
    ctx.set("dicom_dir", dicom_dir)
    result = stage.run(ctx)

    nifti_paths = result.require("nifti_paths")
    assert len(nifti_paths) == 1
    assert nifti_paths[0].exists()
    assert nifti_paths[0].suffix == ".gz"

    metadata = result.require("series_metadata")
    assert len(metadata) == 1
    assert metadata[0]["modality"] == "CT"


def test_dicom_to_nifti_handles_multiple_series(tmp_path: Path) -> None:
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()
    (dicom_dir / "series_a").mkdir()
    (dicom_dir / "series_b").mkdir()
    make_dicom_series(dicom_dir / "series_a", num_slices=3)
    make_dicom_series(dicom_dir / "series_b", num_slices=3)

    output_dir = tmp_path / "nifti"
    stage = DicomToNiftiStage(DicomToNiftiConfig(output_dir=str(output_dir)))

    ctx = PipelineContext()
    ctx.set("dicom_dir", dicom_dir)
    result = stage.run(ctx)

    assert len(result.require("nifti_paths")) == 2


def test_dicom_to_nifti_volume_has_expected_shape(tmp_path: Path) -> None:
    import SimpleITK as sitk

    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()
    make_dicom_series(dicom_dir, num_slices=5, rows=32, columns=24)

    output_dir = tmp_path / "nifti"
    stage = DicomToNiftiStage(DicomToNiftiConfig(output_dir=str(output_dir)))

    ctx = PipelineContext()
    ctx.set("dicom_dir", dicom_dir)
    result = stage.run(ctx)

    image = sitk.ReadImage(str(result.require("nifti_paths")[0]))
    assert image.GetSize() == (24, 32, 5)
