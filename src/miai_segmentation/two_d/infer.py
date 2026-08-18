"""Sliding-window inference for a trained 2D segmentation model.

Structurally identical to
:mod:`miai_segmentation.three_d.infer` -- same sliding-window +
threshold + write-back-with-source-metadata approach -- just with a
2D (``roi_size: tuple[int, int]``) window instead of a 3D one. Kept as
its own module (rather than importing three_d's) because
:class:`InferenceConfig` is a distinct, modality-specific public type
(``roi_size``'s arity differs), per `docs/api_design.md`'s "Package
public surface" section.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

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
            predicted foreground.
        device: ``"cpu"`` or ``"cuda"``.
    """

    roi_size: tuple[int, int] = (256, 256)
    sw_batch_size: int = 1
    overlap: float = 0.25
    threshold: float = 0.5
    device: str = "cpu"


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
            probs = torch.sigmoid(raw_output)
            mask = (probs > config.threshold).squeeze(0).squeeze(0).to(torch.uint8).cpu().numpy()

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
