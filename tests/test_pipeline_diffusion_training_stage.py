"""Integration test for the concrete DiffusionTrainingStage."""

from pathlib import Path

import pytest

from conftest import make_offset_cube_volume
from miai_diffusion.model import DiffusionUNetConfig
from miai_diffusion.schedule import NoiseScheduleConfig
from miai_diffusion.train import DiffusionTrainingConfig
from miai_pipeline.context import PipelineContext
from miai_pipeline.exceptions import StageError
from miai_pipeline.stages.diffusion_training import (
    DiffusionTrainingStage,
    DiffusionTrainingStageConfig,
)
from miai_transforms.config import TransformConfig, TransformSpec

_UNET_CONFIG = DiffusionUNetConfig(
    in_channels=1, base_channels=4, channel_multipliers=(1, 2), time_embedding_dim=16
)
_TRANSFORMS = TransformConfig(
    transforms=[
        TransformSpec(name="load_image", params={"keys": ["image"]}),
        TransformSpec(name="ensure_type", params={"keys": ["image"]}),
    ]
)


@pytest.mark.slow
def test_diffusion_training_stage_writes_checkpoint(tmp_path: Path) -> None:
    image_path = make_offset_cube_volume(tmp_path / "data", name="case0", size=(8, 8, 8))

    ctx = PipelineContext()
    ctx.set("manifest", {"train": [str(image_path)], "val": [], "test": []})

    stage = DiffusionTrainingStage(
        DiffusionTrainingStageConfig(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            transforms=_TRANSFORMS,
            unet=_UNET_CONFIG,
            schedule=NoiseScheduleConfig(num_timesteps=20),
            training=DiffusionTrainingConfig(max_epochs=1, device="cpu"),
        )
    )

    result = stage.run(ctx)

    checkpoint_path = Path(result.require("diffusion_checkpoint_path"))
    assert checkpoint_path.exists()


def test_diffusion_training_stage_empty_train_split_raises(tmp_path: Path) -> None:
    ctx = PipelineContext()
    ctx.set("manifest", {"train": [], "val": [], "test": []})

    stage = DiffusionTrainingStage(
        DiffusionTrainingStageConfig(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            transforms=_TRANSFORMS,
            unet=_UNET_CONFIG,
        )
    )

    with pytest.raises(StageError):
        stage.run(ctx)
