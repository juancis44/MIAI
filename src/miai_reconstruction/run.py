"""Simulating and reconstructing k-space for a list of volumes on disk."""

from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk
import torch

from miai_core.io import ensure_dir
from miai_core.logging import get_logger
from miai_reconstruction.exceptions import ReconstructionError
from miai_reconstruction.kspace import (
    KSpaceReconstructionConfig,
    UndersamplingConfig,
    apply_undersampling,
    build_undersampling_mask,
    reconstruct_from_kspace,
    simulate_kspace,
)

logger = get_logger(__name__)


def run_kspace_reconstruction(
    source_paths: list[str],
    reconstruction_config: KSpaceReconstructionConfig,
    undersampling_config: UndersamplingConfig | None,
    output_dir: str,
) -> list[Path]:
    """Simulate and reconstruct k-space for a list of volumes on disk.

    For each source volume: simulates its k-space (via
    :func:`~miai_reconstruction.kspace.simulate_kspace`), optionally
    zeroes out lines per ``undersampling_config`` (zero-filled
    undersampled reconstruction), then reconstructs an image from that
    k-space and writes it as NIfTI. With ``undersampling_config=None``
    this is a (near-)identity round trip, useful as a correctness
    check of the FFT reconstruction itself; passing an
    :class:`~miai_reconstruction.kspace.UndersamplingConfig`
    demonstrates the actual reconstruction problem -- recovering an
    image from incomplete k-space.

    Args:
        source_paths: Volumes to simulate k-space for and reconstruct.
        reconstruction_config: FFT normalization parameters.
        undersampling_config: If set, k-space lines are zeroed out
            before reconstruction to simulate an accelerated
            acquisition. If ``None``, reconstruction uses the full
            (fully-sampled) simulated k-space.
        output_dir: Directory reconstructed volumes are written to
            (created if missing).

    Returns:
        One reconstructed file path per entry in ``source_paths``, in
        order.

    Raises:
        ReconstructionError: If ``source_paths`` is empty.
    """
    if not source_paths:
        raise ReconstructionError("source_paths is empty; nothing to reconstruct.")

    out_dir = ensure_dir(output_dir)
    reconstructed_paths: list[Path] = []

    for source_path in source_paths:
        reference_image = sitk.ReadImage(str(source_path))
        array = sitk.GetArrayFromImage(reference_image).astype("float32")
        image_tensor = torch.from_numpy(array)

        kspace = simulate_kspace(image_tensor, reconstruction_config)
        if undersampling_config is not None:
            mask = build_undersampling_mask(tuple(image_tensor.shape), undersampling_config)
            kspace = apply_undersampling(kspace, mask)

        reconstructed = reconstruct_from_kspace(kspace, reconstruction_config)
        reconstructed_array = reconstructed.cpu().numpy()

        reconstructed_image = sitk.GetImageFromArray(reconstructed_array)
        reconstructed_image.CopyInformation(reference_image)

        stem = Path(source_path).name.removesuffix(".nii.gz").removesuffix(".nii")
        out_path = out_dir / f"{stem}_reconstructed.nii.gz"
        sitk.WriteImage(reconstructed_image, str(out_path))
        reconstructed_paths.append(out_path)
        logger.info("Wrote reconstructed volume for %s to %s", source_path, out_path)

    return reconstructed_paths
