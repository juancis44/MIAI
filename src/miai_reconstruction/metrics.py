"""Reconstruction-quality metrics: PSNR and SSIM.

Unlike :mod:`miai_evaluation` (Dice / Hausdorff distance, for
comparing segmentation masks), reconstruction quality is a photometric
comparison between a reference image and a reconstructed one, for
which PSNR and SSIM are the standard metrics -- hence a separate
metrics module here rather than extending :mod:`miai_evaluation`,
and the new ``scikit-image`` dependency this module (only) needs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from miai_reconstruction.exceptions import ReconstructionError


def reconstruction_quality(
    reference: npt.NDArray[Any],
    reconstructed: npt.NDArray[Any],
    *,
    win_size: int | None = None,
) -> dict[str, float]:
    """Compute PSNR and SSIM between a reference and reconstructed image.

    Args:
        reference: The ground-truth image/volume array.
        reconstructed: The reconstructed image/volume array, same
            shape as ``reference``.
        win_size: Sliding window size for SSIM, forwarded to
            :func:`skimage.metrics.structural_similarity`. Must be odd
            and no larger than the smallest array dimension; leave as
            ``None`` to use scikit-image's default (7), which requires
            every dimension to be at least 7.

    Returns:
        ``{"psnr": ..., "ssim": ...}``, both as plain ``float``.

    Raises:
        ReconstructionError: If ``reference`` and ``reconstructed`` do
            not have the same shape.
    """
    if reference.shape != reconstructed.shape:
        raise ReconstructionError(
            f"reference shape {reference.shape} does not match "
            f"reconstructed shape {reconstructed.shape}."
        )

    reference = np.asarray(reference, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    data_range = float(reference.max() - reference.min())

    psnr = float(peak_signal_noise_ratio(reference, reconstructed, data_range=data_range))
    ssim = float(
        structural_similarity(reference, reconstructed, data_range=data_range, win_size=win_size)
    )
    return {"psnr": psnr, "ssim": ssim}
