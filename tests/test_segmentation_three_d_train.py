"""Tests for miai_segmentation.three_d.train (tiny real tensors, CPU only)."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureTyped

from conftest import make_synthetic_multiclass_volume_pair, make_synthetic_volume_pair
from miai_segmentation.exceptions import SegmentationError
from miai_segmentation.three_d.models import UNetConfig, build_unet
from miai_segmentation.three_d.train import TrainingConfig, train_model
from miai_transforms.sitk_transforms import LoadImageSitkd

_UNET_CONFIG = UNetConfig(channels=(4, 8), strides=(2,), num_res_units=0)
_MULTICLASS_UNET_CONFIG = UNetConfig(channels=(4, 8), strides=(2,), num_res_units=0, out_channels=4)
_TRANSFORMS = Compose(
    [
        LoadImageSitkd(keys=["image", "label"]),
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


def test_train_model_passes_weight_decay_to_optimizer(tmp_path: Path) -> None:
    """weight_decay is a new TrainingConfig field -- confirm it actually
    reaches torch.optim.Adam (not just accepted and ignored)."""
    train_loader = _make_loader(tmp_path / "train", 1)
    model = build_unet(_UNET_CONFIG)
    config = TrainingConfig(max_epochs=1, device="cpu", weight_decay=0.01)

    with patch("torch.optim.Adam", wraps=torch.optim.Adam) as mock_adam:
        train_model(model, train_loader, None, config, str(tmp_path / "ckpt"))

    mock_adam.assert_called_once()
    assert mock_adam.call_args.kwargs["weight_decay"] == pytest.approx(0.01)


def _dice_on_loader(model: torch.nn.Module, loader: DataLoader) -> float:
    """Compute mean Dice of ``model``'s (thresholded) predictions on ``loader``.

    Mirrors train_model's own validation-scoring logic, but lives here
    (not imported from miai_segmentation.three_d.train) so this test measures
    actual segmentation quality through a path independent of the
    training loop's internal bookkeeping.
    """
    model.eval()
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post = Compose([AsDiscrete(threshold=0.5)])
    with torch.no_grad():
        for batch in loader:
            outputs = torch.sigmoid(model(batch["image"]))
            preds = [post(i) for i in decollate_batch(outputs)]
            labels = [post(i) for i in decollate_batch(batch["label"])]
            dice_metric(y_pred=preds, y=labels)
    aggregated = dice_metric.aggregate()
    metric_tensor = aggregated[0] if isinstance(aggregated, tuple) else aggregated
    dice_metric.reset()
    return float(metric_tensor.item())


@pytest.mark.slow
def test_train_model_actually_learns_to_segment(tmp_path: Path) -> None:
    """Trains for enough epochs on an easy, learnable synthetic pattern
    (make_synthetic_volume_pair's centered cube) and checks the
    resulting model's Dice is both high in absolute terms and clearly
    better than an untrained model of the same architecture -- unlike
    test_train_model_writes_checkpoint, which only checks training runs
    without error and produces a loadable checkpoint, never that the
    model actually learned anything.
    """
    train_loader = _make_loader(tmp_path / "train", n_cases=2)
    val_loader = _make_loader(tmp_path / "val", n_cases=1)

    model = build_unet(_UNET_CONFIG)
    config = TrainingConfig(max_epochs=40, learning_rate=1e-2, val_interval=1, device="cpu")
    checkpoint_path = train_model(
        model, train_loader, val_loader, config, str(tmp_path / "checkpoints")
    )

    trained_model = build_unet(_UNET_CONFIG)
    trained_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    trained_dice = _dice_on_loader(trained_model, val_loader)

    untrained_dice = _dice_on_loader(build_unet(_UNET_CONFIG), val_loader)

    assert trained_dice > 0.5
    assert trained_dice > untrained_dice


def _make_multiclass_loader(tmp_path: Path, n_cases: int, num_classes: int = 4) -> DataLoader:
    data = []
    for i in range(n_cases):
        image_path, label_path = make_synthetic_multiclass_volume_pair(
            tmp_path, name=f"case{i}", num_classes=num_classes
        )
        data.append({"image": str(image_path), "label": str(label_path)})
    dataset = Dataset(data=data, transform=_TRANSFORMS)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


@pytest.mark.slow
def test_train_model_multiclass_writes_checkpoint(tmp_path: Path) -> None:
    """``num_classes > 1`` switches to the softmax/argmax path end to end."""
    train_loader = _make_multiclass_loader(tmp_path / "train", 2)
    val_loader = _make_multiclass_loader(tmp_path / "val", 1)

    model = build_unet(_MULTICLASS_UNET_CONFIG)
    config = TrainingConfig(max_epochs=2, val_interval=1, device="cpu", num_classes=4)

    checkpoint_path = train_model(
        model, train_loader, val_loader, config, str(tmp_path / "checkpoints")
    )

    assert checkpoint_path.exists()
    fresh_model = build_unet(_MULTICLASS_UNET_CONFIG)
    fresh_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))


def _multiclass_dice_on_loader(
    model: torch.nn.Module, loader: DataLoader, num_classes: int
) -> float:
    """Mirrors train_model's multi-class validation scoring, computed independently."""
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=num_classes)])
    post_label = Compose([AsDiscrete(to_onehot=num_classes)])
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["image"])
            preds = [post_pred(i) for i in decollate_batch(outputs)]
            labels = [post_label(i) for i in decollate_batch(batch["label"])]
            dice_metric(y_pred=preds, y=labels)
    aggregated = dice_metric.aggregate()
    metric_tensor = aggregated[0] if isinstance(aggregated, tuple) else aggregated
    dice_metric.reset()
    return float(metric_tensor.item())


@pytest.mark.slow
def test_train_model_multiclass_actually_learns_to_segment(tmp_path: Path) -> None:
    """Same intent as test_train_model_actually_learns_to_segment, multi-class."""
    train_loader = _make_multiclass_loader(tmp_path / "train", n_cases=2)
    val_loader = _make_multiclass_loader(tmp_path / "val", n_cases=1)

    model = build_unet(_MULTICLASS_UNET_CONFIG)
    config = TrainingConfig(
        max_epochs=60, learning_rate=1e-2, val_interval=1, device="cpu", num_classes=4
    )
    checkpoint_path = train_model(
        model, train_loader, val_loader, config, str(tmp_path / "checkpoints")
    )

    trained_model = build_unet(_MULTICLASS_UNET_CONFIG)
    trained_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    trained_dice = _multiclass_dice_on_loader(trained_model, val_loader, num_classes=4)

    untrained_dice = _multiclass_dice_on_loader(
        build_unet(_MULTICLASS_UNET_CONFIG), val_loader, num_classes=4
    )

    assert trained_dice > 0.3
    assert trained_dice > untrained_dice
