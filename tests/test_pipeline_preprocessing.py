"""Tests for miai_pipeline.stages.preprocessing."""

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from miai_pipeline.context import PipelineContext
from miai_pipeline.stages.preprocessing import PreprocessingConfig, PreprocessingStage


def _write_synthetic_volume(
    path: Path,
    *,
    size: tuple[int, int, int] = (20, 20, 10),
    spacing: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> None:
    array = (
        np.random.default_rng(0).normal(loc=100.0, scale=20.0, size=size[::-1]).astype(np.float32)
    )
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    sitk.WriteImage(image, str(path))


def test_preprocessing_resamples_to_target_spacing(tmp_path: Path) -> None:
    volume_path = tmp_path / "volume.nii.gz"
    _write_synthetic_volume(volume_path, size=(20, 20, 10), spacing=(0.5, 0.5, 0.5))

    output_dir = tmp_path / "preprocessed"
    stage = PreprocessingStage(
        PreprocessingConfig(
            output_dir=str(output_dir), target_spacing=(1.0, 1.0, 1.0), normalization="none"
        )
    )

    ctx = PipelineContext()
    ctx.set("nifti_paths", [volume_path])
    result = stage.run(ctx)

    preprocessed_paths = result.require("preprocessed_paths")
    assert len(preprocessed_paths) == 1

    image = sitk.ReadImage(str(preprocessed_paths[0]))
    assert image.GetSpacing() == (1.0, 1.0, 1.0)
    # Halving resolution from 0.5mm to 1.0mm spacing halves the size per axis.
    assert image.GetSize() == (10, 10, 5)


def test_preprocessing_zscore_normalizes_intensity(tmp_path: Path) -> None:
    volume_path = tmp_path / "volume.nii.gz"
    _write_synthetic_volume(volume_path)

    output_dir = tmp_path / "preprocessed"
    stage = PreprocessingStage(
        PreprocessingConfig(
            output_dir=str(output_dir), target_spacing=(0.5, 0.5, 0.5), normalization="zscore"
        )
    )

    ctx = PipelineContext()
    ctx.set("nifti_paths", [volume_path])
    result = stage.run(ctx)

    image = sitk.ReadImage(str(result.require("preprocessed_paths")[0]))
    array = sitk.GetArrayFromImage(image)
    assert abs(float(array.mean())) < 1e-3
    assert abs(float(array.std()) - 1.0) < 1e-2


def test_preprocessing_minmax_normalizes_to_unit_range(tmp_path: Path) -> None:
    volume_path = tmp_path / "volume.nii.gz"
    _write_synthetic_volume(volume_path)

    output_dir = tmp_path / "preprocessed"
    stage = PreprocessingStage(
        PreprocessingConfig(
            output_dir=str(output_dir), target_spacing=(0.5, 0.5, 0.5), normalization="minmax"
        )
    )

    ctx = PipelineContext()
    ctx.set("nifti_paths", [volume_path])
    result = stage.run(ctx)

    image = sitk.ReadImage(str(result.require("preprocessed_paths")[0]))
    array = sitk.GetArrayFromImage(image)
    assert float(array.min()) == 0.0
    assert abs(float(array.max()) - 1.0) < 1e-6


def test_preprocessing_none_normalization_preserves_values(tmp_path: Path) -> None:
    volume_path = tmp_path / "volume.nii.gz"
    _write_synthetic_volume(volume_path, size=(8, 8, 8), spacing=(1.0, 1.0, 1.0))

    output_dir = tmp_path / "preprocessed"
    stage = PreprocessingStage(
        PreprocessingConfig(
            output_dir=str(output_dir), target_spacing=(1.0, 1.0, 1.0), normalization="none"
        )
    )

    ctx = PipelineContext()
    ctx.set("nifti_paths", [volume_path])
    result = stage.run(ctx)

    original = sitk.GetArrayFromImage(sitk.ReadImage(str(volume_path)))
    processed = sitk.GetArrayFromImage(sitk.ReadImage(str(result.require("preprocessed_paths")[0])))
    np.testing.assert_allclose(original, processed, rtol=1e-4)


def test_normalize_unknown_normalization_raises() -> None:
    # normalization is a Literal["zscore", "minmax", "none"], so this
    # branch is unreachable through normal (Pydantic-validated)
    # construction -- model_construct bypasses that validation to
    # exercise the defensive fallback directly, the same way a config
    # loaded via model_construct elsewhere in the codebase could.
    config = PreprocessingConfig.model_construct(
        output_dir="unused",
        target_spacing=(1.0, 1.0, 1.0),
        interpolation="linear",
        normalization="not_a_real_normalization",
    )
    stage = PreprocessingStage(config)

    array = np.zeros((4, 4, 4), dtype=np.float32)
    image = sitk.GetImageFromArray(array)

    with pytest.raises(ValueError, match="Unknown normalization"):
        stage._normalize(image)
