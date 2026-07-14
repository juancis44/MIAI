"""Integration test for the concrete InferenceStage (tiny real inference, CPU only)."""

from pathlib import Path

import pytest
import torch

from conftest import make_synthetic_volume_pair
from miai_pipeline.context import PipelineContext
from miai_pipeline.exceptions import StageError
from miai_pipeline.stages.inference import InferenceStage, InferenceStageConfig
from miai_segmentation.infer import InferenceConfig
from miai_segmentation.models import UNetConfig, build_unet
from miai_transforms.config import TransformConfig, TransformSpec

_UNET_CONFIG = UNetConfig(channels=(4, 8), strides=(2,), num_res_units=0)
_IMAGE_TRANSFORMS = TransformConfig(
    transforms=[
        TransformSpec(name="load_image", params={"keys": ["image"]}),
        TransformSpec(name="ensure_type", params={"keys": ["image"], "dtype": torch.float32}),
    ]
)


@pytest.mark.slow
def test_inference_stage_writes_predictions_using_context_checkpoint(tmp_path: Path) -> None:
    image_path, _ = make_synthetic_volume_pair(tmp_path / "data", size=(16, 16, 16))
    checkpoint_path = tmp_path / "model.pt"
    torch.save(build_unet(_UNET_CONFIG).state_dict(), checkpoint_path)

    ctx = PipelineContext()
    ctx.set("manifest", {"train": [], "val": [], "test": [str(image_path)]})
    ctx.set("model_checkpoint_path", str(checkpoint_path))

    stage = InferenceStage(
        InferenceStageConfig(
            output_dir=str(tmp_path / "predictions"),
            transforms=_IMAGE_TRANSFORMS,
            unet=_UNET_CONFIG,
            inference=InferenceConfig(roi_size=(16, 16, 16), sw_batch_size=1, device="cpu"),
        )
    )

    result = stage.run(ctx)

    prediction_paths = result.require("prediction_paths")
    assert len(prediction_paths) == 1
    assert Path(prediction_paths[0]).exists()


@pytest.mark.slow
def test_inference_stage_explicit_checkpoint_path_overrides_context(tmp_path: Path) -> None:
    image_path, _ = make_synthetic_volume_pair(tmp_path / "data", size=(16, 16, 16))
    checkpoint_path = tmp_path / "explicit_model.pt"
    torch.save(build_unet(_UNET_CONFIG).state_dict(), checkpoint_path)

    ctx = PipelineContext()
    ctx.set("manifest", {"train": [], "val": [], "test": [str(image_path)]})
    # No model_checkpoint_path in context -- config.checkpoint_path must be used.

    stage = InferenceStage(
        InferenceStageConfig(
            output_dir=str(tmp_path / "predictions"),
            transforms=_IMAGE_TRANSFORMS,
            unet=_UNET_CONFIG,
            inference=InferenceConfig(roi_size=(16, 16, 16), sw_batch_size=1, device="cpu"),
            checkpoint_path=str(checkpoint_path),
        )
    )

    result = stage.run(ctx)
    assert len(result.require("prediction_paths")) == 1


def test_inference_stage_empty_test_split_raises(tmp_path: Path) -> None:
    ctx = PipelineContext()
    ctx.set("manifest", {"train": [], "val": [], "test": []})

    stage = InferenceStage(
        InferenceStageConfig(
            output_dir=str(tmp_path / "predictions"),
            transforms=_IMAGE_TRANSFORMS,
            unet=_UNET_CONFIG,
        )
    )

    with pytest.raises(StageError):
        stage.run(ctx)
