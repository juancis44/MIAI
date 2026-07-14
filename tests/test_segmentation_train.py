"""Tests for miai_segmentation.train (tiny real tensors, CPU only)."""

from pathlib import Path

import pytest
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged

from conftest import make_synthetic_volume_pair
from miai_segmentation.exceptions import SegmentationError
from miai_segmentation.models import UNetConfig, build_unet
from miai_segmentation.train import TrainingConfig, train_model

_UNET_CONFIG = UNetConfig(channels=(4, 8), strides=(2,), num_res_units=0)
_TRANSFORMS = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ]
)


def _make_loader(tmp_path: Path, n_cases: int) -> DataLoader:
    data = []
    for i in range(n_cases):
        image_path, label_path = make_synthetic_volume_pair(tmp_path, name=f"case{i}")
        data.append({"image": str(image_path), "label": str(label_path)})
    dataset = Dataset(data=data, transform=_TRANSFORMS)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


@pytest.mark.slow
def test_train_model_writes_checkpoint(tmp_path: Path) -> None:
    train_loader = _make_loader(tmp_path / "train", 2)
    val_loader = _make_loader(tmp_path / "val", 1)

    model = build_unet(_UNET_CONFIG)
    config = TrainingConfig(max_epochs=2, val_interval=1, device="cpu")

    checkpoint_path = train_model(
        model, train_loader, val_loader, config, str(tmp_path / "checkpoints")
    )

    assert checkpoint_path.exists()
    assert checkpoint_path.name == config.checkpoint_name

    fresh_model = build_unet(_UNET_CONFIG)
    fresh_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))


@pytest.mark.slow
def test_train_model_without_val_loader_checkpoints_final_epoch(tmp_path: Path) -> None:
    train_loader = _make_loader(tmp_path / "train", 1)
    model = build_unet(_UNET_CONFIG)
    config = TrainingConfig(max_epochs=1, device="cpu")

    checkpoint_path = train_model(model, train_loader, None, config, str(tmp_path / "ckpt"))
    assert checkpoint_path.exists()


def test_train_model_empty_loader_raises(tmp_path: Path) -> None:
    empty_loader = DataLoader(Dataset(data=[], transform=_TRANSFORMS), batch_size=1)
    model = build_unet(_UNET_CONFIG)
    config = TrainingConfig(max_epochs=1, device="cpu")

    with pytest.raises(SegmentationError):
        train_model(model, empty_loader, None, config, str(tmp_path / "unused"))
