"""Sliding-window inference for a trained segmentation model."""

from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk
import torch
from monai.data import DataLoader
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
        threshold: Sigmoid probability threshold above which a voxel is
            predicted foreground.
        device: ``"cpu"`` or ``"cuda"``.
    """

    roi_size: tuple[int, int, int] = (96, 96, 96)
    sw_batch_size: int = 1
    overlap: float = 0.25
    threshold: float = 0.5
    device: str = "cpu"


def run_inference(
    model: torch.nn.Module,
    data_loader: DataLoader,
    source_paths: list[str],
    checkpoint_path: str,
    config: InferenceConfig,
    output_dir: str,
) -> list[Path]:
    """Run sliding-window inference and write predictions to disk.

    Loads ``checkpoint_path`` into ``model``, then for each batch in
    ``data_loader`` (expected batch size 1 -- see
    :class:`~miai_pipeline.stages.inference.InferenceStage`, which
    forces this) runs :func:`monai.inferers.sliding_window_inference`,
    thresholds the sigmoid output, and writes the result as a NIfTI
    volume that copies the spatial metadata (spacing/origin/direction)
    of the corresponding entry in ``source_paths``.

    The prediction array is transposed from MONAI's array convention
    (``(W, H, D)``, matching the NIfTI affine's axis order, as returned
    by its nibabel/ITK-backed readers) to SimpleITK's ``(D, H, W)``
    convention before being wrapped with
    :func:`SimpleITK.GetImageFromArray`. This is only valid when the
    transform pipeline does not reorder spatial axes between the source
    file and the model's output (e.g. no ``orientation`` transform) --
    true of MIAI's reference workflow, where resampling/orientation is
    already handled upstream by
    :class:`~miai_pipeline.stages.preprocessing.PreprocessingStage`. A
    transform pipeline that reorients axes will produce a spatially
    incorrect prediction file under this function.

    Args:
        model: An untrained model with the same architecture used to
            produce ``checkpoint_path`` (e.g. via
            :func:`miai_segmentation.models.build_unet`).
        data_loader: Yields one-case batches with an ``"image"`` key.
        source_paths: The original (pre-transform) image file path for
            each item in ``data_loader``, in the same order -- used to
            copy spatial metadata into the prediction and to name the
            output file.
        checkpoint_path: Path to a state dict saved by
            :func:`miai_segmentation.train.train_model`.
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
            logits = sliding_window_inference(
                inputs=inputs,
                roi_size=config.roi_size,
                sw_batch_size=config.sw_batch_size,
                predictor=model,
                overlap=config.overlap,
            )
            probs = torch.sigmoid(logits)
            mask_whd = (
                (probs > config.threshold).squeeze(0).squeeze(0).to(torch.uint8).cpu().numpy()
            )
            mask_dhw = mask_whd.transpose(2, 1, 0)

            reference_path = source_paths[idx]
            reference_image = sitk.ReadImage(str(reference_path))
            prediction_image = sitk.GetImageFromArray(mask_dhw)
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
