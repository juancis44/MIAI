"""Training loop for a MIAI 3D segmentation model, binary or multi-class.

Dimension-agnostic in practice (it only calls ``model(inputs)`` and
scores the result), but lives under :mod:`miai_segmentation.three_d`
alongside the architectures it is meant to train -- the 2D/2.5D
modalities may need their own variants later (e.g. 2.5D reassembling
per-slice batches into a volume before computing Dice).

Binary by default (``TrainingConfig.num_classes = 1``): sigmoid logits,
``DiceLoss(sigmoid=True)``, 0.5-threshold post-processing -- unchanged
from this module's original behavior. Setting ``num_classes`` above 1
switches every step (loss, post-processing, and the validation Dice
used for checkpoint selection) to a softmax/argmax multi-class path;
see :class:`TrainingConfig` for the details.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
from monai.data import decollate_batch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose

from miai_core.config import MIAIBaseConfig
from miai_core.io import ensure_dir
from miai_core.logging import get_logger
from miai_segmentation.exceptions import SegmentationError

logger = get_logger(__name__)


class TrainingConfig(MIAIBaseConfig):
    """Configuration for :func:`train_model`.

    Attributes:
        max_epochs: Number of training epochs.
        learning_rate: Adam optimizer learning rate. With ``cosine_
            annealing`` off (the default), this is the single, constant
            rate used for the whole run -- unchanged from this config's
            original behavior. With it on, this is the schedule's
            *starting* (maximum) rate.
        cosine_annealing: If ``True``, wrap the optimizer in
            :class:`torch.optim.lr_scheduler.CosineAnnealingLR`,
            stepped once per epoch, decaying smoothly from
            ``learning_rate`` down to ``min_learning_rate`` over
            ``max_epochs`` (``T_max=max_epochs``, ``eta_min=
            min_learning_rate``) following a cosine curve, rather than
            holding one constant rate for the whole run. ``False`` (the
            default) disables it, unchanged from this config's
            original behavior. If training stops early (see
            ``early_stopping_patience``), the schedule simply stops
            partway through its curve rather than completing it -- the
            same way a fixed epoch budget cut short by early stopping
            always behaves.
        min_learning_rate: The schedule's floor rate (``eta_min``),
            only used when ``cosine_annealing`` is ``True``. Together
            with ``learning_rate`` (the schedule's ceiling), these are
            the *two* rates a cosine-annealed run is actually
            configured by -- one alone does not describe the schedule.
            Defaults to ``0.0``, MONAI/PyTorch's own default for
            ``CosineAnnealingLR``'s ``eta_min``.
        weight_decay: Adam optimizer L2 weight decay (MONAI/PyTorch's
            ``Adam(weight_decay=...)``). ``0.0`` (the default) disables
            it, unchanged from this config's original behavior --
            matches :class:`torch.optim.Adam`'s own default. A nonzero
            value penalizes large weights during training, a standard
            regularizer against overfitting -- complementary to
            :attr:`~miai_segmentation.three_d.models.UNetConfig.dropout`
            /:attr:`~miai_segmentation.two_d.models.UNetConfig.dropout`,
            which regularizes activations rather than weights.
        val_interval: Run validation every ``val_interval`` epochs.
        early_stopping_patience: Stop training early if validation Dice
            has not improved for this many consecutive *validation
            checks* (not epochs -- with ``val_interval > 1`` a patience
            of ``5`` tolerates ``5 * val_interval`` epochs without
            improvement, not 5). ``None`` (the default) disables early
            stopping, unchanged from this config's original
            behavior -- training always runs the full ``max_epochs``.
            Has no effect when ``val_loader`` is ``None`` (there is
            nothing to check patience against). A concrete use case:
            raising ``max_epochs`` well above what a fixed budget would
            safely allow, then letting early stopping cut the run short
            once validation Dice plateaus -- cheaper than guessing the
            "right" epoch count up front, and unlike a fixed higher
            budget, it does not waste compute training past the point
            where the checkpoint stops improving.
        device: ``"cpu"`` or ``"cuda"`` (or a specific CUDA device
            string, e.g. ``"cuda:0"``).
        checkpoint_name: Filename for the best-validation-Dice
            checkpoint, written under ``train_model``'s
            ``checkpoint_dir`` argument.
        num_classes: Number of segmentation classes, including
            background. ``1`` (the default) trains a binary model --
            sigmoid logits, ``DiceLoss(sigmoid=True)``, and 0.5-
            threshold post-processing, unchanged from this function's
            original, binary-only behavior. Any value ``> 1`` switches
            to a multi-class path: softmax logits, ``DiceLoss(softmax
            =True, to_onehot_y=True)`` (the model's ``out_channels``
            must equal ``num_classes``, and ``batch["label"]`` must be
            a single-channel integer class map with values in
            ``[0, num_classes)``, not already one-hot), argmax +
            one-hot post-processing for both predictions and labels,
            and a validation Dice metric that excludes the background
            channel (``include_background=False``) -- so checkpoint
            selection is driven by how well the model segments the
            actual structures of interest, not by the (typically much
            larger, so numerically dominant) background class.
        class_weights: Optional per-channel weight passed straight
            through to :class:`monai.losses.DiceLoss`'s own ``weight``
            argument -- scales each output channel's contribution to
            the training loss, so a structure that is small/hard (and
            would otherwise be outweighed by easier or larger ones) can
            be given more influence over the gradient without changing
            anything else about the loss (still Dice, still on the same
            channels validation Dice is computed over). ``None`` (the
            default) is passed straight through as ``weight=None``,
            MONAI's own default -- every channel weighted equally,
            unchanged from this config's original behavior. When set,
            its length must equal the number of channels the loss
            actually sees: ``num_classes`` for the multi-class path
            (index 0 is background, since ``include_background=True``
            for the loss even though the *validation metric* excludes
            it -- see ``num_classes`` above), or ``1`` for the binary
            path. A mismatched length raises :class:`SegmentationError`
            immediately, before any training happens.
        gradient_clip_norm: Optional maximum gradient L2 norm, passed
            straight through to :func:`torch.nn.utils.clip_grad_norm_`
            (called on every parameter with a gradient, right after
            ``loss.backward()`` and before ``optimizer.step()``) --
            rescales the gradient in place if its norm exceeds this
            value, leaving it unchanged otherwise. ``None`` (the
            default) skips the call entirely, unchanged from this
            config's original behavior. A common use for an unstable
            training run (large, occasional gradient spikes that throw
            off an otherwise-improving optimization trajectory): capping
            the norm bounds how large a single update can be, without
            otherwise changing the loss, learning rate, or optimizer.
    """

    max_epochs: int = 100
    learning_rate: float = 1e-4
    cosine_annealing: bool = False
    min_learning_rate: float = 0.0
    weight_decay: float = 0.0
    val_interval: int = 1
    early_stopping_patience: int | None = None
    device: str = "cpu"
    checkpoint_name: str = "best_model.pt"
    num_classes: int = 1
    class_weights: tuple[float, ...] | None = None
    gradient_clip_norm: float | None = None


def train_model(
    model: torch.nn.Module,
    train_loader: Iterable[dict[str, torch.Tensor]],
    val_loader: Iterable[dict[str, torch.Tensor]] | None,
    config: TrainingConfig,
    checkpoint_dir: str,
) -> Path:
    """Train a segmentation model and checkpoint the best epoch.

    Runs a standard supervised loop: :class:`monai.losses.DiceLoss`
    (sigmoid, or softmax + one-hot for multi-class -- see
    :attr:`TrainingConfig.num_classes`), Adam optimization (at a
    constant rate, or cosine-annealed between ``TrainingConfig.
    learning_rate`` and ``TrainingConfig.min_learning_rate`` -- see
    :attr:`TrainingConfig.cosine_annealing`), and -- if ``val_loader``
    is given -- validation every ``config.val_interval``
    epochs scored with :class:`monai.metrics.DiceMetric`. The
    checkpoint with the highest validation Dice is kept; without a
    validation loader, the final epoch's weights are checkpointed
    instead.

    Args:
        model: The model to train (e.g. from
            :func:`miai_segmentation.three_d.models.build_model`).
        train_loader: Any iterable of batches with ``"image"`` and
            ``"label"`` keys (typically a
            :class:`torch.utils.data.DataLoader`, but any iterable
            works -- this function only ever iterates over it once per
            epoch, nothing DataLoader-specific).
        val_loader: Optional validation loader with the same batch
            shape as ``train_loader``. If ``None``, validation is
            skipped and the last epoch's weights are checkpointed.
        config: Training hyperparameters.
        checkpoint_dir: Directory the checkpoint is written to (created
            if missing).

    Returns:
        Path to the saved checkpoint (a ``torch.save``d state dict).

    Raises:
        SegmentationError: If ``train_loader`` yields no batches.
    """
    device = torch.device(config.device)
    model = model.to(device)

    multiclass = config.num_classes > 1
    expected_weight_channels = config.num_classes if multiclass else 1
    if config.class_weights is not None and len(config.class_weights) != expected_weight_channels:
        raise SegmentationError(
            f"class_weights has {len(config.class_weights)} entries, but the loss "
            f"has {expected_weight_channels} channel(s) "
            f"({'num_classes' if multiclass else 'binary, num_classes=1'})."
        )

    if multiclass:
        loss_function = DiceLoss(
            to_onehot_y=True, softmax=True, include_background=True, weight=config.class_weights
        )
        dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        post_pred = Compose([AsDiscrete(argmax=True, to_onehot=config.num_classes)])
        post_label = Compose([AsDiscrete(to_onehot=config.num_classes)])
    else:
        loss_function = DiceLoss(sigmoid=True, include_background=True, weight=config.class_weights)
        dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        post_pred = Compose([AsDiscrete(threshold=0.5)])
        post_label = Compose([AsDiscrete(threshold=0.5)])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.max_epochs, eta_min=config.min_learning_rate
        )
        if config.cosine_annealing
        else None
    )

    out_dir = ensure_dir(checkpoint_dir)
    checkpoint_path = out_dir / config.checkpoint_name

    best_metric = -1.0
    saw_any_batch = False
    epochs_without_improvement = 0

    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            saw_any_batch = True
            n_batches += 1
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            if config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            epoch_loss += loss.item()

        if n_batches > 0:
            logger.info(
                "Epoch %d/%d - train loss: %.4f",
                epoch + 1,
                config.max_epochs,
                epoch_loss / n_batches,
            )

        if scheduler is not None:
            scheduler.step()
            logger.info(
                "Epoch %d/%d - learning rate: %.6f",
                epoch + 1,
                config.max_epochs,
                optimizer.param_groups[0]["lr"],
            )

        if val_loader is not None and (epoch + 1) % config.val_interval == 0:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch["image"].to(device)
                    labels = batch["label"].to(device)
                    raw_outputs = model(inputs)
                    # Multi-class: argmax is monotonic under softmax, so
                    # post_pred's AsDiscrete(argmax=True) can operate
                    # directly on raw logits -- no need to apply softmax
                    # first. Binary: post_pred thresholds a probability,
                    # so sigmoid must be applied here, as before.
                    outputs = raw_outputs if multiclass else torch.sigmoid(raw_outputs)
                    outputs_list = [post_pred(i) for i in decollate_batch(outputs)]
                    labels_list = [post_label(i) for i in decollate_batch(labels)]
                    dice_metric(y_pred=outputs_list, y=labels_list)

                aggregated = dice_metric.aggregate()
                metric_tensor = aggregated[0] if isinstance(aggregated, tuple) else aggregated
                metric = metric_tensor.item()
                dice_metric.reset()
                logger.info("Epoch %d/%d - val Dice: %.4f", epoch + 1, config.max_epochs, metric)

                if metric > best_metric:
                    best_metric = metric
                    epochs_without_improvement = 0
                    torch.save(model.state_dict(), checkpoint_path)
                    logger.info(
                        "New best val Dice %.4f - checkpoint saved to %s",
                        metric,
                        checkpoint_path,
                    )
                else:
                    epochs_without_improvement += 1

                if (
                    config.early_stopping_patience is not None
                    and epochs_without_improvement >= config.early_stopping_patience
                ):
                    logger.info(
                        "Early stopping at epoch %d/%d - no val Dice improvement in "
                        "%d consecutive validation checks (patience=%d)",
                        epoch + 1,
                        config.max_epochs,
                        epochs_without_improvement,
                        config.early_stopping_patience,
                    )
                    break

    if not saw_any_batch:
        raise SegmentationError("train_loader yielded no batches; cannot train.")

    if not checkpoint_path.exists():
        # No validation loader was given (or it never improved past -1.0):
        # checkpoint whatever the final epoch produced instead.
        torch.save(model.state_dict(), checkpoint_path)

    return checkpoint_path
