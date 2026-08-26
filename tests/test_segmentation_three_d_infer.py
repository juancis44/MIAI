"""Tests for miai_segmentation.three_d.infer (tiny real tensors, CPU only)."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import SimpleITK as sitk
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, EnsureTyped

from conftest import make_synthetic_volume_pair
from miai_segmentation.exceptions import SegmentationError
from miai_segmentation.three_d.infer import InferenceConfig, run_inference
from miai_segmentation.three_d.models import UNetConfig, build_unet
from miai_transforms.sitk_transforms import LoadImageSitkd

_UNET_CONFIG = UNetConfig(channels=(4, 8), strides=(2,), num_res_units=0)
_IMAGE_ONLY_TRANSFORMS = Compose(
    [
        LoadImageSitkd(keys=["image"]),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ]
)


@pytest.mark.slow
def test_run_inference_writes_prediction_matching_reference_geometry(tmp_path: Path) -> None:
    image_path, _ = make_synthetic_volume_pair(tmp_path / "data", size=(16, 16, 16))

    model = build_unet(_UNET_CONFIG)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    test_ds = Dataset(data=[{"image": str(image_path)}], transform=_IMAGE_ONLY_TRANSFORMS)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)

    fresh_model = build_unet(_UNET_CONFIG)
    config = InferenceConfig(roi_size=(16, 16, 16), sw_batch_size=1, overlap=0.0, device="cpu")
    prediction_paths = run_inference(
        fresh_model,
        test_loader,
        [str(image_path)],
        str(checkpoint_path),
        config,
        str(tmp_path / "predictions"),
    )

    assert len(prediction_paths) == 1
    assert prediction_paths[0].exists()

    reference_image = sitk.ReadImage(str(image_path))
    prediction_image = sitk.ReadImage(str(prediction_paths[0]))
    assert prediction_image.GetSize() == reference_image.GetSize()
    assert prediction_image.GetSpacing() == reference_image.GetSpacing()

    pred_array = sitk.GetArrayFromImage(prediction_image)
    assert set(pred_array.flatten().tolist()).issubset({0, 1})


def test_run_inference_mismatched_source_paths_raises(tmp_path: Path) -> None:
    model = build_unet(_UNET_CONFIG)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    class _OneItemLoader:
        def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
            yield {"image": torch.zeros(1, 1, 4, 4, 4)}

    with pytest.raises(SegmentationError):
        run_inference(
            build_unet(_UNET_CONFIG),
            _OneItemLoader(),
            [],
            str(checkpoint_path),
            InferenceConfig(roi_size=(4, 4, 4), sw_batch_size=1, device="cpu"),
            str(tmp_path / "out"),
        )


class _TupleOutputModel(torch.nn.Module):
    """A model whose ``forward`` returns a tuple instead of a single
    tensor -- exercises the defensive type check on
    ``sliding_window_inference``'s output (see the identical test for
    ``miai_segmentation.two_d.infer``, which confirmed empirically that
    MONAI passes a non-tensor predictor output straight through)."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the input twice instead of a single prediction tensor."""
        return (x, x)


def test_run_inference_non_tensor_model_output_raises(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.pt"
    torch.save(_TupleOutputModel().state_dict(), checkpoint_path)

    class _OneItemLoader:
        def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
            yield {"image": torch.zeros(1, 1, 4, 4, 4)}

    with pytest.raises(SegmentationError, match="Expected the model to return a single tensor"):
        run_inference(
            _TupleOutputModel(),
            _OneItemLoader(),
            ["case0.nii.gz"],
            str(checkpoint_path),
            InferenceConfig(roi_size=(4, 4, 4), sw_batch_size=1, device="cpu"),
            str(tmp_path / "out"),
        )


def test_run_inference_fewer_items_than_source_paths_raises(tmp_path: Path) -> None:
    # An empty loader against a non-empty source_paths never enters the
    # per-batch loop (so the loop-internal idx >= len(source_paths)
    # check never fires), which is what's needed to reach the post-loop
    # length-mismatch check instead -- mirrors the identical test for
    # miai_segmentation.two_d.infer.
    model = build_unet(_UNET_CONFIG)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    class _EmptyLoader:
        def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
            return iter(())

    with pytest.raises(SegmentationError, match="yielded 0 items but source_paths provided"):
        run_inference(
            build_unet(_UNET_CONFIG),
            _EmptyLoader(),
            ["case0.nii.gz"],
            str(checkpoint_path),
            InferenceConfig(roi_size=(4, 4, 4), sw_batch_size=1, device="cpu"),
            str(tmp_path / "out"),
        )
