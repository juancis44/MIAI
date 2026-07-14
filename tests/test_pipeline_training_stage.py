"""Integration test for the concrete TrainingStage (tiny real training, CPU only)."""

from pathlib import Path

import pytest
import torch

from conftest import make_synthetic_volume_pair
from miai_pipeline.context import PipelineContext
from miai_pipeline.exceptions import StageError
from miai_pipeline.stages.dataset import DatasetConfig, DatasetStage
from miai_pipeline.stages.training import TrainingStage, TrainingStageConfig
from miai_segmentation.models import UNetConfig
from miai_segmentation.train import TrainingConfig
from miai_transforms.config import TransformConfig, TransformSpec

_UNET_CONFIG = UNetConfig(channels=(4, 8), strides=(2,), num_res_units=0)

_LOAD_TRANSFORMS = [
    TransformSpec(name="load_image", params={"keys": ["image", "label"]}),
    TransformSpec(name="ensure_type", params={"keys": ["image", "label"], "dtype": torch.float32}),
]


def _build_manifest_context(tmp_path: Path, n_cases: int = 3) -> PipelineContext:
    images, labels = [], []
    for i in range(n_cases):
        image_path, label_path = make_synthetic_volume_pair(tmp_path / "data", name=f"case{i}")
        images.append(image_path)
        labels.append(label_path)

    dataset_stage = DatasetStage(
        DatasetConfig(
            manifest_path=str(tmp_path / "manifest.json"),
            val_fraction=1 / 3,
            test_fraction=1 / 3,
            seed=1,
            context_key="images",
            label_context_key="labels",
        )
    )
    ctx = PipelineContext()
    ctx.set("images", images)
    ctx.set("labels", labels)
    return dataset_stage.run(ctx)


@pytest.mark.slow
def test_training_stage_writes_model_checkpoint(tmp_path: Path) -> None:
    ctx = _build_manifest_context(tmp_path)

    stage = TrainingStage(
        TrainingStageConfig(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            train_transforms=TransformConfig(transforms=_LOAD_TRANSFORMS),
            val_transforms=TransformConfig(transforms=_LOAD_TRANSFORMS),
            unet=_UNET_CONFIG,
            training=TrainingConfig(max_epochs=1, device="cpu"),
        )
    )

    result = stage.run(ctx)

    checkpoint_path = Path(result.require("model_checkpoint_path"))
    assert checkpoint_path.exists()


def test_training_stage_empty_train_split_raises(tmp_path: Path) -> None:
    ctx = PipelineContext()
    ctx.set("manifest", {"train": [], "val": [], "test": []})

    stage = TrainingStage(
        TrainingStageConfig(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            train_transforms=TransformConfig(transforms=_LOAD_TRANSFORMS),
            val_transforms=TransformConfig(transforms=_LOAD_TRANSFORMS),
            unet=_UNET_CONFIG,
        )
    )

    with pytest.raises(StageError):
        stage.run(ctx)
