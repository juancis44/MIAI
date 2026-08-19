"""Integration test: TrainingStage + InferenceStage in "two_d" modality.

Confirms the slice-expansion wiring (miai_datasets.slices.expand_to_slice_dicts
+ miai_transforms.slice_transforms.ExtractSliced + miai_segmentation.modality)
lets a 2D model train and run inference over whole-NIfTI cases end to end,
producing one reassembled (D, H, W) prediction volume per case -- the same
one-file-per-case contract three_d's modality has always had.
"""

from pathlib import Path

import pytest
import SimpleITK as sitk
import torch

from conftest import make_synthetic_volume_pair
from miai_pipeline.context import PipelineContext
from miai_pipeline.stages.inference import InferenceStage, InferenceStageConfig
from miai_pipeline.stages.training import TrainingStage, TrainingStageConfig
from miai_segmentation.modality import SegmentationInferenceConfig, SegmentationModalityConfig
from miai_segmentation.three_d.train import TrainingConfig
from miai_segmentation.two_d.infer import InferenceConfig
from miai_segmentation.two_d.models import ArchitectureConfig, UNetConfig
from miai_transforms.config import TransformConfig, TransformSpec

_ARCHITECTURE_CONFIG = SegmentationModalityConfig(
    modality="two_d",
    two_d=ArchitectureConfig(
        kind="unet", unet=UNetConfig(channels=(4, 8), strides=(2,), num_res_units=0)
    ),
)

_TRAIN_TRANSFORMS = TransformConfig(
    transforms=[
        TransformSpec(name="load_image", params={"keys": ["image", "label"]}),
        TransformSpec(name="extract_slice", params={"keys": ["image", "label"]}),
        TransformSpec(
            name="ensure_type", params={"keys": ["image", "label"], "dtype": torch.float32}
        ),
    ]
)
_TEST_TRANSFORMS = TransformConfig(
    transforms=[
        TransformSpec(name="load_image", params={"keys": ["image"]}),
        TransformSpec(name="extract_slice", params={"keys": ["image"]}),
        TransformSpec(name="ensure_type", params={"keys": ["image"], "dtype": torch.float32}),
    ]
)


@pytest.mark.slow
def test_two_d_modality_training_and_inference_reassembles_case_volume(tmp_path: Path) -> None:
    train_image, train_label = make_synthetic_volume_pair(
        tmp_path / "train", name="train0", size=(6, 8, 8)
    )
    test_image, _ = make_synthetic_volume_pair(tmp_path / "test", name="test0", size=(6, 8, 8))

    ctx = PipelineContext()
    ctx.set(
        "manifest",
        {
            "train": [{"image": str(train_image), "label": str(train_label)}],
            "val": [],
            "test": [str(test_image)],
        },
    )

    training_stage = TrainingStage(
        TrainingStageConfig(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            train_transforms=_TRAIN_TRANSFORMS,
            val_transforms=_TRAIN_TRANSFORMS,
            architecture=_ARCHITECTURE_CONFIG,
            training=TrainingConfig(max_epochs=1, device="cpu"),
        )
    )
    ctx = training_stage.run(ctx)
    assert Path(ctx.require("model_checkpoint_path")).exists()

    inference_stage = InferenceStage(
        InferenceStageConfig(
            output_dir=str(tmp_path / "predictions"),
            transforms=_TEST_TRANSFORMS,
            architecture=_ARCHITECTURE_CONFIG,
            inference=SegmentationInferenceConfig(
                two_d=InferenceConfig(roi_size=(8, 8), sw_batch_size=1, device="cpu")
            ),
        )
    )
    ctx = inference_stage.run(ctx)

    prediction_paths = ctx.require("prediction_paths")
    assert len(prediction_paths) == 1

    prediction = sitk.ReadImage(str(prediction_paths[0]))
    source = sitk.ReadImage(str(test_image))
    # Reassembled volume must match the source case's full depth/size,
    # not just one slice -- confirms run_case_inference correctly
    # regrouped per-slice predictions back into one case volume.
    assert prediction.GetSize() == source.GetSize()
