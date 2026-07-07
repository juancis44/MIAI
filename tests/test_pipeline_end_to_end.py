"""End-to-end test: DICOM -> NIfTI -> preprocessing -> dataset manifest,
run both stage-by-stage and via a config-driven Pipeline."""

from pathlib import Path

from conftest import make_dicom_series
from miai_pipeline import Pipeline, PipelineConfig, PipelineContext


def _make_two_series(dicom_root: Path) -> None:
    (dicom_root / "series_a").mkdir(parents=True)
    (dicom_root / "series_b").mkdir(parents=True)
    make_dicom_series(dicom_root / "series_a", num_slices=3, rows=16, columns=16)
    make_dicom_series(dicom_root / "series_b", num_slices=3, rows=16, columns=16)


def test_end_to_end_pipeline_via_config(tmp_path: Path) -> None:
    dicom_dir = tmp_path / "dicom"
    _make_two_series(dicom_dir)

    config = PipelineConfig.model_validate(
        {
            "stages": [
                {"type": "dicom_to_nifti", "params": {"output_dir": str(tmp_path / "nifti")}},
                {
                    "type": "preprocessing",
                    "params": {
                        "output_dir": str(tmp_path / "preprocessed"),
                        "target_spacing": [1.0, 1.0, 1.0],
                        "normalization": "zscore",
                    },
                },
                {
                    "type": "dataset",
                    "params": {
                        "manifest_path": str(tmp_path / "manifest.json"),
                        "val_fraction": 0.5,
                        "seed": 1,
                    },
                },
            ]
        }
    )

    pipeline = Pipeline.from_config(config)

    ctx = PipelineContext()
    ctx.set("dicom_dir", dicom_dir)
    result = pipeline.run(ctx)

    manifest = result.require("manifest")
    total_cases = sum(len(v) for v in manifest.values())
    assert total_cases == 2
    assert (tmp_path / "manifest.json").exists()

    for split_paths in manifest.values():
        for p in split_paths:
            assert Path(p).exists()
            assert Path(p).name.endswith("_preprocessed.nii.gz")
