"""End-to-end tests chaining several real pipeline stages together via a
config-driven Pipeline, rather than exercising each stage in isolation
(as the per-stage test_pipeline_*_stage.py files do).

Covers, across the four scenarios below: the full main-workflow data-prep
half (DICOM -> NIfTI -> Preprocessing -> Registration -> Dataset), the
full ML half (Dataset -> Training -> Inference -> Evaluation), and two
optional-stage chains (diffusion training -> denoising, training ->
export) -- specifically checking that context keys one stage writes
(e.g. ``model_checkpoint_path``, ``diffusion_checkpoint_path``) are
picked up automatically by the next stage, without a test manually
injecting them, since a real pipeline run never does that either.
"""

from pathlib import Path
from typing import Any

import numpy.typing as npt
import pytest
import torch

from conftest import make_dicom_series, make_offset_cube_volume, make_synthetic_volume_pair
from miai_foundation_models.extractor import FeatureExtractor
from miai_pipeline import Pipeline, PipelineConfig, PipelineContext

_UNET_PARAMS = {"channels": [4, 8], "strides": [2], "num_res_units": 0}
_ARCHITECTURE_PARAMS = {"kind": "unet", "unet": _UNET_PARAMS}
_DIFFUSION_UNET_PARAMS = {
    "in_channels": 1,
    "base_channels": 4,
    "channel_multipliers": [1, 2],
    "time_embedding_dim": 16,
}


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


def test_end_to_end_pipeline_with_registration(tmp_path: Path) -> None:
    """DICOM -> NIfTI -> Preprocessing -> [Registration] -> Dataset, the
    full main-workflow data-prep half advertised in the root README's
    workflow diagram (previously only tested up to Preprocessing)."""
    dicom_dir = tmp_path / "dicom"
    _make_two_series(dicom_dir)

    fixed_path = make_offset_cube_volume(tmp_path / "atlas", name="atlas", size=(16, 16, 16))

    config = PipelineConfig.model_validate(
        {
            "stages": [
                {"type": "dicom_to_nifti", "params": {"output_dir": str(tmp_path / "nifti")}},
                {
                    "type": "preprocessing",
                    "params": {
                        "output_dir": str(tmp_path / "preprocessed"),
                        "target_spacing": [1.0, 1.0, 1.0],
                        "normalization": "none",
                    },
                },
                {
                    "type": "registration",
                    "params": {
                        "fixed_image_path": str(fixed_path),
                        "output_dir": str(tmp_path / "registered"),
                        "transform_dir": str(tmp_path / "transforms"),
                        "registration": {
                            "transform_type": "rigid",
                            "metric": "mean_squares",
                            "number_of_iterations": 50,
                            "sampling_percentage": 1.0,
                            "shrink_factors": [1],
                            "smoothing_sigmas": [0.0],
                        },
                    },
                },
                {
                    "type": "dataset",
                    "params": {
                        "manifest_path": str(tmp_path / "manifest.json"),
                        "context_key": "registered_paths",
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

    registered_paths = result.require("registered_paths")
    assert len(registered_paths) == 2
    for p in registered_paths:
        assert Path(p).exists()

    manifest = result.require("manifest")
    assert sum(len(v) for v in manifest.values()) == 2


@pytest.mark.slow
def test_end_to_end_ml_workflow_dataset_to_evaluation(tmp_path: Path) -> None:
    """Dataset -> Training -> Inference -> Evaluation, chained as one
    config-driven Pipeline (previously only tested stage-by-stage, each
    starting from a hand-built context rather than a preceding stage's
    real output)."""
    images, labels = [], []
    for i in range(4):
        image_path, label_path = make_synthetic_volume_pair(tmp_path / "data", name=f"case{i}")
        images.append(str(image_path))
        labels.append(str(label_path))

    load_both = {
        "transforms": [
            {"name": "load_image", "params": {"keys": ["image", "label"]}},
            {
                "name": "ensure_type",
                "params": {"keys": ["image", "label"], "dtype": torch.float32},
            },
        ]
    }
    load_image_only = {
        "transforms": [
            {"name": "load_image", "params": {"keys": ["image"]}},
            {"name": "ensure_type", "params": {"keys": ["image"], "dtype": torch.float32}},
        ]
    }

    config = PipelineConfig.model_validate(
        {
            "stages": [
                {
                    "type": "dataset",
                    "params": {
                        "manifest_path": str(tmp_path / "manifest.json"),
                        "context_key": "images",
                        "label_context_key": "labels",
                        "val_fraction": 0.25,
                        "test_fraction": 0.25,
                        "seed": 1,
                    },
                },
                {
                    "type": "training",
                    "params": {
                        "checkpoint_dir": str(tmp_path / "checkpoints"),
                        "train_transforms": load_both,
                        "val_transforms": load_both,
                        "architecture": _ARCHITECTURE_PARAMS,
                        "training": {"max_epochs": 1, "device": "cpu"},
                    },
                },
                {
                    "type": "inference",
                    "params": {
                        "output_dir": str(tmp_path / "predictions"),
                        "transforms": load_image_only,
                        "architecture": _ARCHITECTURE_PARAMS,
                        "inference": {
                            "roi_size": [16, 16, 16],
                            "sw_batch_size": 1,
                            "device": "cpu",
                        },
                    },
                },
                {
                    "type": "evaluation",
                    "params": {"report_path": str(tmp_path / "metrics.json")},
                },
            ]
        }
    )

    pipeline = Pipeline.from_config(config)

    ctx = PipelineContext()
    ctx.set("images", images)
    ctx.set("labels", labels)
    result = pipeline.run(ctx)

    # Confirm each stage really consumed the previous stage's real
    # output rather than something injected by the test.
    assert Path(result.require("model_checkpoint_path")).exists()
    assert len(result.require("prediction_paths")) == 1

    metrics = result.require("metrics")
    assert len(metrics["per_case"]) == 1
    assert "dice" in metrics["mean"]
    assert Path(tmp_path / "metrics.json").exists()


@pytest.mark.slow
def test_end_to_end_diffusion_training_and_denoising(tmp_path: Path) -> None:
    """diffusion_training -> denoising, chained -- denoising picks up
    diffusion_checkpoint_path from context automatically rather than
    the test passing a checkpoint path explicitly."""
    image_path = make_offset_cube_volume(tmp_path / "data", name="case0", size=(8, 8, 8))

    config = PipelineConfig.model_validate(
        {
            "stages": [
                {
                    "type": "diffusion_training",
                    "params": {
                        "checkpoint_dir": str(tmp_path / "checkpoints"),
                        "transforms": {
                            "transforms": [
                                {"name": "load_image", "params": {"keys": ["image"]}},
                                {"name": "ensure_type", "params": {"keys": ["image"]}},
                            ]
                        },
                        "unet": _DIFFUSION_UNET_PARAMS,
                        "schedule": {"num_timesteps": 20},
                        "training": {"max_epochs": 1, "device": "cpu"},
                    },
                },
                {
                    "type": "denoising",
                    "params": {
                        "output_dir": str(tmp_path / "denoised"),
                        "unet": _DIFFUSION_UNET_PARAMS,
                        "schedule": {"num_timesteps": 20},
                        "denoise": {"start_timestep": 5, "device": "cpu"},
                    },
                },
            ]
        }
    )

    pipeline = Pipeline.from_config(config)

    ctx = PipelineContext()
    ctx.set("manifest", {"train": [str(image_path)], "val": [], "test": []})
    ctx.set("preprocessed_paths", [image_path])
    result = pipeline.run(ctx)

    assert Path(result.require("diffusion_checkpoint_path")).exists()
    denoised_paths = result.require("denoised_paths")
    assert len(denoised_paths) == 1
    assert Path(denoised_paths[0]).exists()


@pytest.mark.slow
def test_end_to_end_training_and_export(tmp_path: Path) -> None:
    """training -> export, chained -- export picks up model_checkpoint_path
    from context automatically rather than the test passing a checkpoint
    path explicitly."""
    image0, label0 = make_synthetic_volume_pair(tmp_path / "data", name="case0")
    image1, label1 = make_synthetic_volume_pair(tmp_path / "data", name="case1")

    load_both = {
        "transforms": [
            {"name": "load_image", "params": {"keys": ["image", "label"]}},
            {
                "name": "ensure_type",
                "params": {"keys": ["image", "label"], "dtype": torch.float32},
            },
        ]
    }

    config = PipelineConfig.model_validate(
        {
            "stages": [
                {
                    "type": "training",
                    "params": {
                        "checkpoint_dir": str(tmp_path / "checkpoints"),
                        "train_transforms": load_both,
                        "val_transforms": load_both,
                        "architecture": _ARCHITECTURE_PARAMS,
                        "training": {"max_epochs": 1, "device": "cpu"},
                    },
                },
                {
                    "type": "export",
                    "params": {
                        "output_dir": str(tmp_path / "bundle"),
                        "architecture": _ARCHITECTURE_PARAMS,
                        "export": {
                            "format": "torchscript",
                            "example_input_shape": [1, 1, 16, 16, 16],
                        },
                        "metadata": {"name": "e2e-test-unet", "version": "0.0.1"},
                    },
                },
            ]
        }
    )

    pipeline = Pipeline.from_config(config)

    ctx = PipelineContext()
    ctx.set(
        "manifest",
        {
            "train": [{"image": str(image0), "label": str(label0)}],
            "val": [{"image": str(image1), "label": str(label1)}],
            "test": [],
        },
    )
    result = pipeline.run(ctx)

    assert Path(result.require("model_checkpoint_path")).exists()
    bundle_path = Path(result.require("deploy_bundle_path"))
    assert (bundle_path / "model.pt").exists()
    assert (bundle_path / "metadata.yaml").exists()


@pytest.mark.slow
def test_end_to_end_reconstruction_feature_extraction_visualization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DICOM -> NIfTI -> Preprocessing -> {reconstruction, feature_extraction,
    visualization}, chained as one config-driven Pipeline instead of each
    optional stage being exercised in isolation from a hand-built context
    (as test_pipeline_reconstruction_stage.py, test_pipeline_feature_extraction_stage.py,
    and test_pipeline_visualization_stage.py do). All three read the real
    ``preprocessed_paths`` written by the real PreprocessingStage that
    precedes them, exactly as a real pipeline run would.

    Monkeypatches FeatureExtractor.from_pretrained for the same reason
    test_pipeline_feature_extraction_stage.py does: keeps this test
    CI-hermetic regardless of network access, without weakening the
    context-wiring assertion it's checking.
    """
    embedding_dim = 4

    class _FakeExtractor:
        def extract_volume_embedding(self, volume: npt.NDArray[Any]) -> torch.Tensor:
            return torch.zeros(embedding_dim)

    monkeypatch.setattr(
        FeatureExtractor,
        "from_pretrained",
        classmethod(lambda cls, config: _FakeExtractor()),
    )

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
                    "type": "reconstruction",
                    "params": {
                        "output_dir": str(tmp_path / "reconstructed"),
                        "undersampling": {"acceleration": 4.0},
                    },
                },
                {
                    "type": "feature_extraction",
                    "params": {"output_dir": str(tmp_path / "embeddings")},
                },
                {
                    "type": "visualization",
                    "params": {
                        "output_dir": str(tmp_path / "qc"),
                        "montage": {"num_slices": 2},
                    },
                },
            ]
        }
    )

    pipeline = Pipeline.from_config(config)

    ctx = PipelineContext()
    ctx.set("dicom_dir", dicom_dir)
    result = pipeline.run(ctx)

    # Every optional stage below consumed the real preprocessed_paths
    # written by PreprocessingStage -- confirm each wrote its own output
    # for both cases, rather than something the test injected itself.
    preprocessed_paths = result.require("preprocessed_paths")
    assert len(preprocessed_paths) == 2

    reconstructed_paths = result.require("reconstructed_paths")
    assert len(reconstructed_paths) == 2
    for p in reconstructed_paths:
        assert Path(p).exists()

    embedding_paths = result.require("embedding_paths")
    assert len(embedding_paths) == 2
    for p in embedding_paths:
        assert Path(p).exists()

    qc_paths = result.require("qc_visualization_paths")
    assert len(qc_paths) == 2
    for p in qc_paths:
        assert Path(p).exists()
