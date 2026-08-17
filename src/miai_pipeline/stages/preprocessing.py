"""Resampling and intensity normalization stage."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import SimpleITK as sitk

from miai_core.config import MIAIBaseConfig
from miai_core.io import ensure_dir
from miai_core.logging import get_logger
from miai_pipeline.context import PipelineContext
from miai_pipeline.stage import PipelineStage

logger = get_logger(__name__)

_INTERPOLATORS = {
    "linear": sitk.sitkLinear,
    "nearest": sitk.sitkNearestNeighbor,
    "bspline": sitk.sitkBSpline,
}


class PreprocessingConfig(MIAIBaseConfig):
    """Configuration for :class:`PreprocessingStage`.

    Attributes:
        output_dir: Directory where preprocessed volumes are written.
        target_spacing: Target voxel spacing in millimeters (x, y, z).
        interpolation: Resampling interpolator: ``"linear"`` for
            intensity images, ``"nearest"`` for label maps,
            ``"bspline"`` for smoother intensity resampling.
        normalization: ``"zscore"`` (zero mean, unit variance),
            ``"minmax"`` (rescale to ``[0, 1]``), or ``"none"``.
    """

    output_dir: str
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    interpolation: Literal["linear", "nearest", "bspline"] = "linear"
    normalization: Literal["zscore", "minmax", "none"] = "zscore"


class PreprocessingStage(PipelineStage):
    """Resample volumes to a target spacing and normalize intensities.

    Reads:
        ``nifti_paths`` (``list[Path]``): volumes to preprocess, as
        produced by :class:`~miai_pipeline.stages.dicom_to_nifti.DicomToNiftiStage`.

    Writes:
        ``preprocessed_paths`` (``list[Path]``): the preprocessed
        volumes, in the same order as ``nifti_paths``.
    """

    name = "preprocessing"
    config_cls = PreprocessingConfig

    def __init__(self, config: PreprocessingConfig) -> None:
        """Store this stage's configuration."""
        self.config = config

    def run(self, context: PipelineContext) -> PipelineContext:
        """Run the stage; see the class docstring for its read/write contract."""
        nifti_paths: list[Path] = context.require("nifti_paths")
        output_dir = ensure_dir(self.config.output_dir)

        preprocessed_paths = []
        for path in nifti_paths:
            logger.info("Preprocessing %s", path)
            image = sitk.ReadImage(str(path))
            image = self._resample(image)
            image = self._normalize(image)

            out_path = output_dir / f"{path.stem.removesuffix('.nii')}_preprocessed.nii.gz"
            sitk.WriteImage(image, str(out_path))
            preprocessed_paths.append(out_path)

        context.set("preprocessed_paths", preprocessed_paths)
        return context

    def _resample(self, image: sitk.Image) -> sitk.Image:
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        target_spacing = self.config.target_spacing

        new_size = [
            max(1, round(original_size[i] * (original_spacing[i] / target_spacing[i])))
            for i in range(3)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0)
        resampler.SetInterpolator(_INTERPOLATORS[self.config.interpolation])
        return cast(sitk.Image, resampler.Execute(image))

    def _normalize(self, image: sitk.Image) -> sitk.Image:
        if self.config.normalization == "none":
            return image
        working_image = sitk.Cast(image, sitk.sitkFloat32)
        if self.config.normalization == "zscore":
            return cast(sitk.Image, sitk.Normalize(working_image))
        if self.config.normalization == "minmax":
            return cast(sitk.Image, sitk.RescaleIntensity(working_image, 0.0, 1.0))
        raise ValueError(f"Unknown normalization: {self.config.normalization}")
