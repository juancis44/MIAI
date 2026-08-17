"""DICOM series -> NIfTI conversion stage."""

from __future__ import annotations

import SimpleITK as sitk

from miai_core.config import MIAIBaseConfig
from miai_core.io import ensure_dir
from miai_core.logging import get_logger
from miai_dicom.series import load_series
from miai_pipeline.context import PipelineContext
from miai_pipeline.exceptions import StageError
from miai_pipeline.stage import PipelineStage

logger = get_logger(__name__)


class DicomToNiftiConfig(MIAIBaseConfig):
    """Configuration for :class:`DicomToNiftiStage`.

    Attributes:
        output_dir: Directory where converted ``.nii.gz`` volumes are
            written, one file per DICOM series found.
    """

    output_dir: str


class DicomToNiftiStage(PipelineStage):
    """Convert every DICOM series under a directory into a NIfTI volume.

    Reads:
        ``dicom_dir`` (``str`` or ``Path``): directory containing one or
        more DICOM series (searched recursively).

    Writes:
        ``nifti_paths`` (``list[Path]``): one ``.nii.gz`` file per
        series found, named by ``SeriesInstanceUID``.
        ``series_metadata`` (``list[dict]``): the metadata (from
        :func:`miai_dicom.metadata.extract_metadata`) of each series'
        first file, in the same order as ``nifti_paths``.
    """

    name = "dicom_to_nifti"
    config_cls = DicomToNiftiConfig

    def __init__(self, config: DicomToNiftiConfig) -> None:
        """Store this stage's configuration."""
        self.config = config

    def run(self, context: PipelineContext) -> PipelineContext:
        """Run the stage; see the class docstring for its read/write contract."""
        from miai_dicom.io import read_dicom
        from miai_dicom.metadata import extract_metadata

        dicom_dir = context.require("dicom_dir")
        series_list = load_series(dicom_dir)
        output_dir = ensure_dir(self.config.output_dir)

        nifti_paths = []
        series_metadata = []
        for series in series_list:
            logger.info("Converting series %s (%d files)", series.series_instance_uid, len(series))
            try:
                reader = sitk.ImageSeriesReader()
                reader.SetFileNames([str(p) for p in series.file_paths])
                image = reader.Execute()
            except RuntimeError as exc:
                raise StageError(
                    f"Failed to read DICOM series {series.series_instance_uid} "
                    f"with SimpleITK: {exc}"
                ) from exc

            out_path = output_dir / f"{series.series_instance_uid}.nii.gz"
            sitk.WriteImage(image, str(out_path))
            nifti_paths.append(out_path)
            series_metadata.append(extract_metadata(read_dicom(series.file_paths[0])))

        context.set("nifti_paths", nifti_paths)
        context.set("series_metadata", series_metadata)
        return context
