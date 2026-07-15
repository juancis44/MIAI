"""Tests for reconstruction_quality (PSNR/SSIM)."""

from __future__ import annotations

import numpy as np
import pytest

from miai_reconstruction.exceptions import ReconstructionError
from miai_reconstruction.metrics import reconstruction_quality


def test_identical_images_have_perfect_ssim() -> None:
    rng = np.random.default_rng(0)
    image = rng.random((8, 8, 8)).astype(np.float32)

    result = reconstruction_quality(image, image, win_size=3)

    assert result["ssim"] == pytest.approx(1.0)
    assert result["psnr"] == float("inf") or result["psnr"] > 100


def test_noisy_image_has_lower_scores_than_identical() -> None:
    rng = np.random.default_rng(1)
    reference = rng.random((8, 8, 8)).astype(np.float32)
    noisy = reference + rng.normal(scale=0.5, size=reference.shape).astype(np.float32)

    identical_scores = reconstruction_quality(reference, reference, win_size=3)
    noisy_scores = reconstruction_quality(reference, noisy, win_size=3)

    assert noisy_scores["ssim"] < identical_scores["ssim"]
    assert noisy_scores["psnr"] < identical_scores["psnr"]


def test_shape_mismatch_raises() -> None:
    reference = np.zeros((8, 8, 8), dtype=np.float32)
    reconstructed = np.zeros((4, 4, 4), dtype=np.float32)

    with pytest.raises(ReconstructionError):
        reconstruction_quality(reference, reconstructed)
