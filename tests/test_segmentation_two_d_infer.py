"""Tests for miai_segmentation.two_d.infer (tiny real tensors, CPU only).

Mirrors tests/test_segmentation_three_d_infer.py's pattern for
`run_inference`, then adds coverage for `run_case_inference` --
the slice-reassembly entry point 2D/2.5D pipeline stages use (see
tests/test_pipeline_two_d_modality.py for its happy path via the full
pipeline) -- including the error branches that only exercising the
happy path never reaches: a `case_slice_counts`/`source_paths` length
mismatch, a loader that yields fewer slices than expected, and one that
yields more.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
import SimpleITK as sitk
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, EnsureTyped

from conftest import make_synthetic_volume_pair
from miai_segmentation.exceptions import SegmentationError
from miai_segmentation.two_d.infer import InferenceConfig, run_case_inference, run_inference
from miai_segmentation.two_d.models import UNetConfig, build_unet
from miai_transforms.sitk_transforms import LoadImageSitkd
from miai_transforms.slice_transforms import ExtractSliced

_UNET_CONFIG = UNetConfig(channels=(4, 8), strides=(2,), num_res_units=0)
_MULTICLASS_UNET_CONFIG = UNetConfig(channels=(4, 8), strides=(2,), num_res_units=0, out_channels=4)
# For run_case_inference: a slice-index-driven pipeline over 3D volumes.
_IMAGE_ONLY_TRANSFORMS = Compose(
    [
        LoadImageSitkd(keys=["image"]),
        ExtractSliced(keys=["image"]),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ]
)
# For run_inference: genuinely 2D image files, no slice extraction needed.
_IMAGE_ONLY_TRANSFORMS_2D = Compose(
    [
        LoadImageSitkd(keys=["image"]),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ]
)


class _TupleOutputModel(torch.nn.Module):
    """A model whose ``forward`` returns a tuple instead of a single
    tensor -- exercises `_predict_slice_mask`'s defensive type check
    (real segmentation architectures never do this; a custom/experimental
    model wired in through `miai_segmentation.modality` could)."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the input twice instead of a single prediction tensor."""
        return (x, x)


class _FixedItemLoader:
    """A minimal ``data_loader`` stand-in yielding a fixed number of items.

    Used to exercise `run_case_inference`'s alignment checks without
    needing a real dataset/model round trip for every case.
    """

    def __init__(self, n_items: int, shape: tuple[int, ...] = (1, 1, 4, 4)) -> None:
        self._n_items = n_items
        self._shape = shape

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for _ in range(self._n_items):
            yield {"image": torch.zeros(*self._shape)}


def _make_2d_image(path: Path, size: tuple[int, int] = (16, 16)) -> Path:
    """Write a genuine 2D NIfTI file -- what `run_inference` (unlike
    `run_case_inference`) expects `source_paths` to point at: a dataset
    of actual 2D images, not slices lifted from a 3D volume."""
    import numpy as np

    height, width = size
    array = np.zeros((height, width), dtype=np.float32)
    image = sitk.GetImageFromArray(array)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))
    return path


@pytest.mark.slow
def test_run_inference_writes_prediction_matching_reference_geometry(tmp_path: Path) -> None:
    image_path = _make_2d_image(tmp_path / "data" / "case0.nii.gz", size=(16, 16))

    model = build_unet(_UNET_CONFIG)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    test_ds = Dataset(data=[{"image": str(image_path)}], transform=_IMAGE_ONLY_TRANSFORMS_2D)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)

    fresh_model = build_unet(_UNET_CONFIG)
    config = InferenceConfig(roi_size=(16, 16), sw_batch_size=1, overlap=0.0, device="cpu")
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

    pred_array = sitk.GetArrayFromImage(prediction_image)
    assert set(pred_array.flatten().tolist()).issubset({0, 1})


@pytest.mark.slow
def test_run_inference_multiclass_writes_class_id_mask(tmp_path: Path) -> None:
    """``num_classes > 1`` switches to softmax + argmax, producing an
    integer class-id mask instead of a 0/1 one."""
    image_path = _make_2d_image(tmp_path / "data" / "case0.nii.gz", size=(16, 16))

    model = build_unet(_MULTICLASS_UNET_CONFIG)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    test_ds = Dataset(data=[{"image": str(image_path)}], transform=_IMAGE_ONLY_TRANSFORMS_2D)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)

    fresh_model = build_unet(_MULTICLASS_UNET_CONFIG)
    config = InferenceConfig(
        roi_size=(16, 16), sw_batch_size=1, overlap=0.0, device="cpu", num_classes=4
    )
    prediction_paths = run_inference(
        fresh_model,
        test_loader,
        [str(image_path)],
        str(checkpoint_path),
        config,
        str(tmp_path / "predictions"),
    )

    reference_image = sitk.ReadImage(str(image_path))
    prediction_image = sitk.ReadImage(str(prediction_paths[0]))
    assert prediction_image.GetSize() == reference_image.GetSize()

    pred_array = sitk.GetArrayFromImage(prediction_image)
    assert set(pred_array.flatten().tolist()).issubset({0, 1, 2, 3})


def test_run_inference_mismatched_source_paths_raises(tmp_path: Path) -> None:
    model = build_unet(_UNET_CONFIG)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    with pytest.raises(SegmentationError):
        run_inference(
            build_unet(_UNET_CONFIG),
            _FixedItemLoader(1),
            [],
            str(checkpoint_path),
            InferenceConfig(roi_size=(4, 4), sw_batch_size=1, device="cpu"),
            str(tmp_path / "out"),
        )


def test_run_inference_fewer_items_than_source_paths_raises(tmp_path: Path) -> None:
    model = build_unet(_UNET_CONFIG)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    with pytest.raises(SegmentationError):
        run_inference(
            build_unet(_UNET_CONFIG),
            _FixedItemLoader(0),
            ["case0.nii.gz"],
            str(checkpoint_path),
            InferenceConfig(roi_size=(4, 4), sw_batch_size=1, device="cpu"),
            str(tmp_path / "out"),
        )


def test_run_inference_non_tensor_model_output_raises(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.pt"
    torch.save(_TupleOutputModel().state_dict(), checkpoint_path)

    with pytest.raises(SegmentationError, match="Expected the model to return a single tensor"):
        run_inference(
            _TupleOutputModel(),
            _FixedItemLoader(1),
            ["case0.nii.gz"],
            str(checkpoint_path),
            InferenceConfig(roi_size=(4, 4), sw_batch_size=1, device="cpu"),
            str(tmp_path / "out"),
        )


def test_run_case_inference_mismatched_lengths_raises(tmp_path: Path) -> None:
    model = build_unet(_UNET_CONFIG)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    with pytest.raises(SegmentationError, match="case_slice_counts has"):
        run_case_inference(
            build_unet(_UNET_CONFIG),
            _FixedItemLoader(0),
            [3, 2],
            ["case0.nii.gz"],
            str(checkpoint_path),
            InferenceConfig(roi_size=(4, 4), sw_batch_size=1, device="cpu"),
            str(tmp_path / "out"),
        )


def test_run_case_inference_fewer_slices_than_expected_raises(tmp_path: Path) -> None:
    model = build_unet(_UNET_CONFIG)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    # case_slice_counts says 3 slices are coming, but the loader only has 2.
    with pytest.raises(SegmentationError, match="fewer slices"):
        run_case_inference(
            build_unet(_UNET_CONFIG),
            _FixedItemLoader(2),
            [3],
            ["case0.nii.gz"],
            str(checkpoint_path),
            InferenceConfig(roi_size=(4, 4), sw_batch_size=1, device="cpu"),
            str(tmp_path / "out"),
        )


@pytest.mark.slow
def test_run_case_inference_more_slices_than_expected_raises(tmp_path: Path) -> None:
    image_path, _ = make_synthetic_volume_pair(tmp_path / "data", size=(2, 8, 8))
    model = build_unet(_UNET_CONFIG)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    # 3 slice-level items in the loader, but case_slice_counts only
    # accounts for 2 -- one item is left over once every case is consumed.
    test_ds = Dataset(
        data=[
            {"image": str(image_path), "slice_index": "0"},
            {"image": str(image_path), "slice_index": "1"},
            {"image": str(image_path), "slice_index": "1"},
        ],
        transform=_IMAGE_ONLY_TRANSFORMS,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    with pytest.raises(SegmentationError, match="more slice"):
        run_case_inference(
            build_unet(_UNET_CONFIG),
            test_loader,
            [2],
            [str(image_path)],
            str(checkpoint_path),
            InferenceConfig(roi_size=(8, 8), sw_batch_size=1, overlap=0.0, device="cpu"),
            str(tmp_path / "out"),
        )


@pytest.mark.slow
def test_run_case_inference_reassembles_volume_in_source_paths_order(tmp_path: Path) -> None:
    image0, _ = make_synthetic_volume_pair(tmp_path / "data", name="case0", size=(3, 8, 8))
    image1, _ = make_synthetic_volume_pair(tmp_path / "data", name="case1", size=(2, 8, 8))

    model = build_unet(_UNET_CONFIG)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    test_ds = Dataset(
        data=[
            {"image": str(image0), "slice_index": "0"},
            {"image": str(image0), "slice_index": "1"},
            {"image": str(image0), "slice_index": "2"},
            {"image": str(image1), "slice_index": "0"},
            {"image": str(image1), "slice_index": "1"},
        ],
        transform=_IMAGE_ONLY_TRANSFORMS,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    prediction_paths = run_case_inference(
        build_unet(_UNET_CONFIG),
        test_loader,
        [3, 2],
        [str(image0), str(image1)],
        str(checkpoint_path),
        InferenceConfig(roi_size=(8, 8), sw_batch_size=1, overlap=0.0, device="cpu"),
        str(tmp_path / "out"),
    )

    assert len(prediction_paths) == 2
    depths = [sitk.ReadImage(str(p)).GetSize()[2] for p in prediction_paths]
    assert depths == [3, 2]
