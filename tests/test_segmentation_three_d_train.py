"""Tests for miai_segmentation.three_d.train (tiny real tensors, CPU only)."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.losses import DiceLoss
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


def test_train_model_cosine_annealing_wires_both_learning_rates(tmp_path: Path) -> None:
    """cosine_annealing/min_learning_rate are new TrainingConfig fields
    -- confirm CosineAnnealingLR is actually constructed with both the
    ceiling (learning_rate, via the optimizer it wraps) and the floor
    (min_learning_rate, as eta_min), not just accepted and ignored."""
    train_loader = _make_loader(tmp_path / "train", 1)
    model = build_unet(_UNET_CONFIG)
    config = TrainingConfig(
        max_epochs=3,
        device="cpu",
        learning_rate=0.1,
        cosine_annealing=True,
        min_learning_rate=0.001,
    )

    with (
        patch("torch.optim.Adam", wraps=torch.optim.Adam) as mock_adam,
        patch(
            "torch.optim.lr_scheduler.CosineAnnealingLR",
            wraps=torch.optim.lr_scheduler.CosineAnnealingLR,
        ) as mock_scheduler,
    ):
        train_model(model, train_loader, None, config, str(tmp_path / "ckpt"))

    # learning_rate (the ceiling) reaches the optimizer at construction
    # time, same as it always has -- captured here via call_args rather
    # than by inspecting the optimizer after the fact, since the
    # scheduler mutates its param_groups["lr"] as training proceeds.
    mock_adam.assert_called_once()
    assert mock_adam.call_args.kwargs["lr"] == pytest.approx(0.1)
    # min_learning_rate (the floor) reaches CosineAnnealingLR as eta_min,
    # and max_epochs sets the schedule's full length (T_max).
    mock_scheduler.assert_called_once()
    assert mock_scheduler.call_args.kwargs["T_max"] == config.max_epochs
    assert mock_scheduler.call_args.kwargs["eta_min"] == pytest.approx(0.001)


def test_train_model_cosine_annealing_decays_learning_rate(tmp_path: Path) -> None:
    """End-to-end check (not just the constructor-wiring check above):
    with cosine_annealing on, the optimizer's actual learning rate
    should decay over epochs from learning_rate towards
    min_learning_rate, not stay constant. Captures the real
    CosineAnnealingLR instance train_model constructs internally (via
    a wrapping side_effect) rather than poking at the optimizer
    directly -- PyTorch's own scheduler instruments ``optimizer.step``
    for its own bookkeeping, so overriding it from the test would
    conflict with that instrumentation."""
    train_loader = _make_loader(tmp_path / "train", 1)
    model = build_unet(_UNET_CONFIG)
    config = TrainingConfig(
        max_epochs=10,
        device="cpu",
        learning_rate=0.1,
        cosine_annealing=True,
        min_learning_rate=0.0,
    )
    created_schedulers: list[torch.optim.lr_scheduler.CosineAnnealingLR] = []
    real_cosine_annealing_lr = torch.optim.lr_scheduler.CosineAnnealingLR

    def _capturing_scheduler(
        *args: object, **kwargs: object
    ) -> torch.optim.lr_scheduler.CosineAnnealingLR:
        scheduler = real_cosine_annealing_lr(*args, **kwargs)  # type: ignore[arg-type]
        created_schedulers.append(scheduler)
        return scheduler

    with patch("torch.optim.lr_scheduler.CosineAnnealingLR", side_effect=_capturing_scheduler):
        train_model(model, train_loader, None, config, str(tmp_path / "ckpt"))

    assert len(created_schedulers) == 1
    final_lr = created_schedulers[0].get_last_lr()[0]
    # 10 epochs stepped from learning_rate=0.1 towards min_learning_rate=0.0
    # should land well below the starting rate -- not exactly 0.0 (T_max
    # matches max_epochs exactly, landing at the schedule's very last
    # point rather than past it), but a clear, unambiguous decrease.
    assert final_lr < 0.1 * 0.5


def test_train_model_no_scheduler_by_default_learning_rate_constant(tmp_path: Path) -> None:
    """cosine_annealing defaults to False -- confirm no scheduler is
    even constructed, and the optimizer's rate never changes."""
    train_loader = _make_loader(tmp_path / "train", 1)
    model = build_unet(_UNET_CONFIG)
    config = TrainingConfig(max_epochs=3, device="cpu", learning_rate=0.05)

    with patch(
        "torch.optim.lr_scheduler.CosineAnnealingLR",
        wraps=torch.optim.lr_scheduler.CosineAnnealingLR,
    ) as mock_scheduler:
        train_model(model, train_loader, None, config, str(tmp_path / "ckpt"))

    mock_scheduler.assert_not_called()


class _CountingLoader:
    """Wraps a DataLoader, counting how many times it's iterated over.

    ``train_model`` calls ``for batch in train_loader`` exactly once
    per epoch, so wrapping the real loader and counting ``__iter__``
    calls is a way to observe how many epochs actually ran -- without
    early stopping, that's ``config.max_epochs``; with it, it should be
    fewer whenever the loop breaks before exhausting the budget.
    """

    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader
        self.iterations = 0

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        self.iterations += 1
        return iter(self._loader)


def test_train_model_early_stopping_stops_before_max_epochs(tmp_path: Path) -> None:
    """learning_rate=0.0 freezes the model's weights, so with a
    deterministic (no-dropout) architecture every validation check
    after the first produces the exact same val Dice -- never a strict
    improvement. With early_stopping_patience=2, training should stop
    after 3 epochs (the first checkpointing epoch, plus 2 consecutive
    non-improving checks) rather than running the full max_epochs=10
    budget."""
    train_loader = _CountingLoader(_make_loader(tmp_path / "train", 1))
    val_loader = _make_loader(tmp_path / "val", 1)
    model = build_unet(_UNET_CONFIG)
    config = TrainingConfig(
        max_epochs=10,
        val_interval=1,
        device="cpu",
        learning_rate=0.0,
        early_stopping_patience=2,
    )

    checkpoint_path = train_model(model, train_loader, val_loader, config, str(tmp_path / "ckpt"))

    assert checkpoint_path.exists()
    assert train_loader.iterations == 3
    assert train_loader.iterations < config.max_epochs


def test_train_model_no_early_stopping_by_default_runs_full_budget(tmp_path: Path) -> None:
    """Same plateaued-Dice setup as the early-stopping test above, but
    with early_stopping_patience left at its default (None) -- training
    should run the full max_epochs budget regardless of how many
    validation checks in a row show no improvement."""
    train_loader = _CountingLoader(_make_loader(tmp_path / "train", 1))
    val_loader = _make_loader(tmp_path / "val", 1)
    model = build_unet(_UNET_CONFIG)
    config = TrainingConfig(max_epochs=4, val_interval=1, device="cpu", learning_rate=0.0)

    train_model(model, train_loader, val_loader, config, str(tmp_path / "ckpt"))

    assert train_loader.iterations == config.max_epochs


def test_train_model_early_stopping_has_no_effect_without_val_loader(tmp_path: Path) -> None:
    """early_stopping_patience is meaningless with no validation loader
    to check patience against -- training should still run the full
    max_epochs budget."""
    train_loader = _CountingLoader(_make_loader(tmp_path / "train", 1))
    model = build_unet(_UNET_CONFIG)
    config = TrainingConfig(max_epochs=3, device="cpu", early_stopping_patience=1)

    train_model(model, train_loader, None, config, str(tmp_path / "ckpt"))

    assert train_loader.iterations == config.max_epochs


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


def test_train_model_class_weights_reach_dice_loss(tmp_path: Path) -> None:
    """class_weights is a new TrainingConfig field -- confirm it actually
    reaches monai.losses.DiceLoss's own ``weight`` argument (not just
    accepted and ignored), for the multi-class path."""
    train_loader = _make_multiclass_loader(tmp_path / "train", 1)
    model = build_unet(_MULTICLASS_UNET_CONFIG)
    weights = (0.1, 2.0, 3.0, 4.0)
    config = TrainingConfig(max_epochs=1, device="cpu", num_classes=4, class_weights=weights)

    with patch("miai_segmentation.three_d.train.DiceLoss", wraps=DiceLoss) as mock_dice_loss:
        train_model(model, train_loader, None, config, str(tmp_path / "ckpt"))

    mock_dice_loss.assert_called_once()
    assert mock_dice_loss.call_args.kwargs["weight"] == weights


def test_train_model_class_weights_default_none_unweighted(tmp_path: Path) -> None:
    """class_weights defaults to None -- confirm DiceLoss is still
    constructed with weight=None (MONAI's own default, every channel
    weighted equally), unchanged from this config's original behavior."""
    train_loader = _make_loader(tmp_path / "train", 1)
    model = build_unet(_UNET_CONFIG)
    config = TrainingConfig(max_epochs=1, device="cpu")

    with patch("miai_segmentation.three_d.train.DiceLoss", wraps=DiceLoss) as mock_dice_loss:
        train_model(model, train_loader, None, config, str(tmp_path / "ckpt"))

    mock_dice_loss.assert_called_once()
    assert mock_dice_loss.call_args.kwargs["weight"] is None


def test_train_model_class_weights_wrong_length_raises_before_training(
    tmp_path: Path,
) -> None:
    """A class_weights length mismatched with the loss's actual channel
    count (num_classes for multi-class, 1 for binary) should fail fast
    with a clear error, before any batch is even iterated."""
    train_loader = _CountingLoader(_make_multiclass_loader(tmp_path / "train", 1))
    model = build_unet(_MULTICLASS_UNET_CONFIG)
    # 3 weights given, but num_classes=4 (background + RV + Myo + LV).
    config = TrainingConfig(
        max_epochs=1, device="cpu", num_classes=4, class_weights=(1.0, 2.0, 3.0)
    )

    with pytest.raises(SegmentationError):
        train_model(model, train_loader, None, config, str(tmp_path / "ckpt"))

    assert train_loader.iterations == 0


def test_train_model_class_weights_wrong_length_binary_raises(tmp_path: Path) -> None:
    """Same fail-fast check, binary path: class_weights must have exactly
    1 entry there (num_classes defaults to 1)."""
    train_loader = _make_loader(tmp_path / "train", 1)
    model = build_unet(_UNET_CONFIG)
    config = TrainingConfig(max_epochs=1, device="cpu", class_weights=(1.0, 2.0))

    with pytest.raises(SegmentationError):
        train_model(model, train_loader, None, config, str(tmp_path / "ckpt"))


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
