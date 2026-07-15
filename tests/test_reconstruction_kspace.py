"""Unit tests for k-space simulation, reconstruction, and undersampling."""

from __future__ import annotations

import pytest
import torch

from miai_reconstruction.exceptions import ReconstructionError
from miai_reconstruction.kspace import (
    KSpaceReconstructionConfig,
    UndersamplingConfig,
    apply_undersampling,
    build_undersampling_mask,
    reconstruct_from_kspace,
    simulate_kspace,
)


def test_roundtrip_reconstruction_matches_original() -> None:
    torch.manual_seed(0)
    image = torch.rand(8, 8, 8)

    kspace = simulate_kspace(image)
    reconstructed = reconstruct_from_kspace(kspace)

    assert torch.allclose(image, reconstructed, atol=1e-4)


def test_reconstruct_from_kspace_returns_real_valued_tensor() -> None:
    image = torch.rand(8, 8, 8)
    kspace = simulate_kspace(image)

    reconstructed = reconstruct_from_kspace(kspace)

    assert not reconstructed.is_complex()


def test_build_undersampling_mask_keeps_center_fully_sampled() -> None:
    config = UndersamplingConfig(acceleration=4.0, center_fraction=0.5, axis=-1)
    mask = build_undersampling_mask((8, 8), config)

    center_start = (8 - 4) // 2
    assert torch.all(mask[..., center_start : center_start + 4] == 1.0)


def test_build_undersampling_mask_acceleration_below_one_raises() -> None:
    config = UndersamplingConfig(acceleration=0.5)

    with pytest.raises(ReconstructionError):
        build_undersampling_mask((8, 8), config)


def test_apply_undersampling_zeros_masked_lines() -> None:
    kspace = torch.ones(4, 4, dtype=torch.complex64)
    mask = torch.zeros(4, 4)

    result = apply_undersampling(kspace, mask)

    assert torch.all(result == 0)


def test_undersampled_reconstruction_differs_from_full() -> None:
    torch.manual_seed(1)
    image = torch.rand(16, 16, 16)
    config = KSpaceReconstructionConfig()
    kspace = simulate_kspace(image, config)

    full_reconstruction = reconstruct_from_kspace(kspace, config)

    mask = build_undersampling_mask(
        tuple(kspace.shape), UndersamplingConfig(acceleration=8.0, center_fraction=0.08, seed=0)
    )
    undersampled_reconstruction = reconstruct_from_kspace(apply_undersampling(kspace, mask), config)

    assert not torch.allclose(full_reconstruction, undersampled_reconstruction, atol=1e-3)
