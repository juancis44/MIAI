"""Sliding-window inference for a trained 2D segmentation model.

Structurally identical to
:mod:`miai_segmentation.three_d.infer` -- same sliding-window +
threshold-or-argmax + write-back-with-source-metadata approach -- just
with a 2D (``roi_size: tuple[int, int]``) window instead of a 3D one.
Binary or multi-class, per :attr:`InferenceConfig.num_classes`. Kept as
its own module (rather than importing three_d's) because
:class:`InferenceConfig` is a distinct, modality-specific public type
(``roi_size``'s arity differs), per `docs/api_design.md`'s "Package
public surface" section.

Two entry points, for two different data shapes:

- :func:`run_inference`: one ``data_loader`` item in, one prediction
  file out -- for a loader that already yields whole 2D cases (a
  dataset genuinely made of individual images, not slices of a volume).
- :func:`run_case_inference`: ``data_loader`` yields per-**slice**
  items (as built from
  :func:`miai_datasets.slices.expand_to_slice_dicts`); this reassembles
  each case's slices back into one prediction volume, so a 2D or 2.5D
  model can still plug into
  :class:`~miai_pipeline.stages.inference.InferenceStage`'s
  one-file-per-case contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import torch
from monai.inferers import sliding_window_inference

from miai_core.config import MIAIBaseConfig
from miai_core.io import ensure_dir
from miai_core.logging import get_logger
from miai_segmentation.exceptions import SegmentationError

logger = get_logger(__name__)


class InferenceConfig(MIAIBaseConfig):
    """Configuration for :func:`run_inference`.

    Attributes:
        roi_size: Sliding-window patch size (must match, or be
            compatible with, the model's expected input size).
        sw_batch_size: Number of windows evaluated per forward pass.
        overlap: Fractional overlap between adjacent windows.
        threshold: Sigmoid probability threshold above which a pixel is
            predicted foreground. Ignored when ``num_classes > 1``
            (multi-class prediction uses argmax instead, which needs no
            threshold).
        device: ``"cpu"`` or ``"cuda"``.
        num_classes: Number of segmentation classes, including
            background. ``1`` (the default) is the original binary
            path: sigmoid probabilities, thresholded at ``threshold``.
            Any value ``> 1`` switches to softmax + argmax: each pixel
            is assigned the class with the highest softmax probability,
            producing an integer class-id mask with values in
            ``[0, num_classes)`` instead of a 0/1 mask. Must match the
            ``out_channels`` of the model that produced ``checkpoint_
            path``.
    """

    roi_size: tuple[int, int] = (256, 256)
    sw_batch_size: int = 1
    overlap: float = 0.25
    threshold: float = 0.5
    device: str = "cpu"
    num_classes: int = 1


def _predict_slice_mask(
    model: torch.nn.Module, inputs: torch.Tensor, config: InferenceConfig
) -> Any:
    """Run one sliding-window forward pass and reduce it to a mask array.

    Shared by :func:`run_inference` and :func:`run_case_inference` --
    identical per-slice prediction logic; only what happens to the
    resulting mask (write it directly vs. accumulate it into a volume)
    differs between them. Binary (``config.num_classes == 1``):
    sigmoid + threshold, a 0/1 mask, unchanged from this function's
    original behavior. Multi-class (``config.num_classes > 1``):
    softmax + argmax, an integer class-id mask in
    ``[0, num_classes)`` -- argmax is monotonic under softmax, so the
    softmax step is skipped, matching :func:`~miai_segmentation.three_d
    .train.train_model`'s validation-time reasoning for the same
    simplification.
    """
    raw_output = sliding_window_inference(
        inputs=inputs,
        roi_size=config.roi_size,
        sw_batch_size=config.sw_batch_size,
        predictor=model,
        overlap=config.overlap,
    )
    if not isinstance(raw_output, torch.Tensor):
        raise SegmentationError(
            "Expected the model to return a single tensor from "
            f"sliding_window_inference, got {type(raw_output).__name__}."
        )
    if config.num_classes > 1:
        mask = raw_output.argmax(dim=1).squeeze(0).to(torch.uint8).cpu().numpy()
    else:
        probs = torch.sigmoid(raw_output)
        mask = (probs > config.threshold).squeeze(0).squeeze(0).to(torch.uint8).cpu().numpy()
    return mask


def run_inference(
    model: torch.nn.Module,
    data_loader: Iterable[dict[str, torch.Tensor]],
    source_paths: list[str],
    checkpoint_path: str,
    config: InferenceConfig,
    output_dir: str,
) -> list[Path]:
    """Run sliding-window inference and write predictions to disk.

    Loads ``checkpoint_path`` into ``model``, then for each batch in
    ``data_loader`` (expected batch size 1) runs
    :func:`monai.inferers.sliding_window_inference`, thresholds the
    sigmoid output, and writes the result as an image that copies the
    spatial metadata (spacing/origin/direction) of the corresponding
    entry in ``source_paths``. See
    :func:`miai_segmentation.three_d.infer.run_inference` for the
    equivalent 3D behavior this mirrors.

    Args:
        model: An untrained model with the same architecture used to
            produce ``checkpoint_path`` (e.g. via
            :func:`miai_segmentation.two_d.models.build_model`).
        data_loader: Any iterable of one-case batches with an
            ``"image"`` key.
        source_paths: The original (pre-transform) image file path for
            each item in ``data_loader``, in the same order -- used to
            copy spatial metadata into the prediction and to name the
            output file.
        checkpoint_path: Path to a state dict saved by
            :func:`miai_segmentation.two_d.train.train_model`.
        config: Sliding-window inference parameters.
        output_dir: Directory predictions are written to (created if
            missing).

    Returns:
        One prediction file path per item in ``data_loader``, in order.

    Raises:
        SegmentationError: If ``data_loader`` yields a different number
            of items than ``source_paths`` provides.
    """
    device = torch.device(config.device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    out_dir = ensure_dir(output_dir)
    prediction_paths: list[Path] = []

    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            if idx >= len(source_paths):
                raise SegmentationError(
                    "data_loader yielded more items than source_paths provided "
                    f"({len(source_paths)})."
                )
            inputs = batch["image"].to(device)
            mask = _predict_slice_mask(model, inputs, config)

            reference_path = source_paths[idx]
            reference_image = sitk.ReadImage(str(reference_path))
            prediction_image = sitk.GetImageFromArray(mask)
            prediction_image.CopyInformation(reference_image)

            stem = Path(reference_path).name.removesuffix(".nii.gz").removesuffix(".nii")
            out_path = out_dir / f"{stem}_pred.nii.gz"
            sitk.WriteImage(prediction_image, str(out_path))
            prediction_paths.append(out_path)
            logger.info("Wrote prediction for %s to %s", reference_path, out_path)

    if len(prediction_paths) != len(source_paths):
        raise SegmentationError(
            f"data_loader yielded {len(prediction_paths)} items but "
            f"source_paths provided {len(source_paths)}."
        )

    return prediction_paths


def run_case_inference(
    model: torch.nn.Module,
    data_loader: Iterable[dict[str, torch.Tensor]],
    case_slice_counts: list[int],
    source_paths: list[str],
    checkpoint_path: str,
    config: InferenceConfig,
    output_dir: str,
) -> list[Path]:
    """Run per-slice inference and reassemble each case's slices into one volume.

    Unlike :func:`run_inference` (which writes one output per
    ``data_loader`` item, 1:1 against ``source_paths``), this expects
    ``data_loader`` to yield one item **per slice** -- as produced by
    building a dataset over
    :func:`miai_datasets.slices.expand_to_slice_dicts`'s output, in
    case-major, ascending-``slice_index`` order (the order that
    function guarantees) -- and regroups them: ``case_slice_counts[i]``
    consecutive items are consumed and stacked (in the order consumed,
    which matches ascending depth) into a single ``(D, H, W)`` mask
    volume for case ``i``, then written out exactly like
    :func:`~miai_segmentation.three_d.infer.run_inference` does (one
    file per case, copying ``source_paths[i]``'s spatial metadata). This
    is what lets a 2D or 2.5D model plug into
    :class:`~miai_pipeline.stages.inference.InferenceStage`'s
    one-prediction-volume-per-case contract, unchanged from the 3D case.

    Args:
        model: An untrained model with the same architecture used to
            produce ``checkpoint_path``.
        data_loader: An iterable yielding one-slice batches with an
            ``"image"`` key, batch size 1, in case-major ascending-slice
            order -- see
            :class:`~miai_pipeline.stages.inference.InferenceStage` for
            how this is built in practice (batch size 1 and no
            shuffling are both required for the regrouping below to be
            correct).
        case_slice_counts: Number of slices contributed by each case, in
            the same order as ``source_paths`` -- from
            :func:`miai_datasets.slices.expand_to_slice_dicts`.
        source_paths: The original (pre-transform, whole-volume) image
            file path for each **case** (not each slice) -- used to
            copy spatial metadata into each case's reassembled
            prediction and to name its output file.
        checkpoint_path: Path to a state dict saved by
            :func:`miai_segmentation.two_d.train.train_model`.
        config: Sliding-window inference parameters (2D window).
        output_dir: Directory predictions are written to (created if
            missing).

    Returns:
        One prediction file path per case, in ``source_paths`` order.

    Raises:
        SegmentationError: If ``case_slice_counts`` and ``source_paths``
            have different lengths, or ``data_loader`` yields fewer or
            more slices in total than ``case_slice_counts`` sums to.
    """
    if len(case_slice_counts) != len(source_paths):
        raise SegmentationError(
            f"case_slice_counts has {len(case_slice_counts)} entries but source_paths "
            f"has {len(source_paths)}; they must be aligned one count per case."
        )

    device = torch.device(config.device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    out_dir = ensure_dir(output_dir)
    prediction_paths: list[Path] = []
    loader_iter: Iterator[dict[str, torch.Tensor]] = iter(data_loader)

    with torch.no_grad():
        for case_idx, n_slices in enumerate(case_slice_counts):
            slice_masks = []
            for _ in range(n_slices):
                try:
                    batch = next(loader_iter)
                except StopIteration as exc:
                    raise SegmentationError(
                        "data_loader yielded fewer slices than case_slice_counts "
                        f"sums to (expected {sum(case_slice_counts)} total)."
                    ) from exc
                inputs = batch["image"].to(device)
                slice_masks.append(_predict_slice_mask(model, inputs, config))

            volume_mask = np.stack(slice_masks, axis=0)

            reference_path = source_paths[case_idx]
            reference_image = sitk.ReadImage(str(reference_path))
            prediction_image = sitk.GetImageFromArray(volume_mask)
            prediction_image.CopyInformation(reference_image)

            stem = Path(reference_path).name.removesuffix(".nii.gz").removesuffix(".nii")
            out_path = out_dir / f"{stem}_pred.nii.gz"
            sitk.WriteImage(prediction_image, str(out_path))
            prediction_paths.append(out_path)
            logger.info(
                "Wrote %d-slice reassembled prediction for %s to %s",
                n_slices,
                reference_path,
                out_path,
            )

    _unexpected = object()
    if next(loader_iter, _unexpected) is not _unexpected:
        raise SegmentationError(
            "data_loader yielded more slice(s) than case_slice_counts accounted for."
        )

    return prediction_paths
