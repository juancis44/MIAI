"""Tests for miai_pipeline.stages.dataset."""

import json
from pathlib import Path

import pytest

from miai_core.exceptions import ConfigError
from miai_pipeline.context import PipelineContext
from miai_pipeline.stages.dataset import DatasetConfig, DatasetStage


def _cases(n: int) -> list[Path]:
    return [Path(f"/data/case_{i}.nii.gz") for i in range(n)]


def test_dataset_split_sizes(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    stage = DatasetStage(
        DatasetConfig(
            manifest_path=str(manifest_path),
            val_fraction=0.2,
            test_fraction=0.1,
            seed=0,
            context_key="preprocessed_paths",
        )
    )

    ctx = PipelineContext()
    ctx.set("preprocessed_paths", _cases(20))
    result = stage.run(ctx)

    manifest = result.require("manifest")
    assert len(manifest["test"]) == 2
    assert len(manifest["val"]) == 4
    assert len(manifest["train"]) == 14
    assert manifest_path.exists()
    assert json.loads(manifest_path.read_text()) == manifest


def test_dataset_split_is_reproducible_with_same_seed(tmp_path: Path) -> None:
    cases = _cases(10)

    def run_once(path: Path):
        stage = DatasetStage(DatasetConfig(manifest_path=str(path), seed=7))
        ctx = PipelineContext()
        ctx.set("preprocessed_paths", cases)
        return stage.run(ctx).require("manifest")

    first = run_once(tmp_path / "manifest1.json")
    second = run_once(tmp_path / "manifest2.json")

    assert first == second


def test_dataset_no_test_split_by_default(tmp_path: Path) -> None:
    stage = DatasetStage(DatasetConfig(manifest_path=str(tmp_path / "manifest.json")))
    ctx = PipelineContext()
    ctx.set("preprocessed_paths", _cases(10))

    manifest = stage.run(ctx).require("manifest")

    assert manifest["test"] == []


def test_dataset_rejects_fractions_summing_to_one_or_more(tmp_path: Path) -> None:
    stage = DatasetStage(
        DatasetConfig(
            manifest_path=str(tmp_path / "manifest.json"),
            val_fraction=0.6,
            test_fraction=0.4,
        )
    )
    ctx = PipelineContext()
    ctx.set("preprocessed_paths", _cases(10))

    with pytest.raises(ConfigError):
        stage.run(ctx)


def test_dataset_uses_configured_context_key(tmp_path: Path) -> None:
    stage = DatasetStage(
        DatasetConfig(
            manifest_path=str(tmp_path / "manifest.json"),
            context_key="nifti_paths",
        )
    )
    ctx = PipelineContext()
    ctx.set("nifti_paths", _cases(5))

    manifest = stage.run(ctx).require("manifest")

    total = len(manifest["train"]) + len(manifest["val"]) + len(manifest["test"])
    assert total == 5
