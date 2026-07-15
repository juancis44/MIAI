"""MRI reconstruction from k-space data.

MIAI's reference reconstruction task is MRI k-space reconstruction:
converting frequency-domain ("k-space") data to an image via the
inverse Fourier transform, and (for the more realistic case) recovering
an image from *undersampled* k-space, the way accelerated MRI
acquisitions work in practice. Implemented entirely with
:mod:`torch.fft`, so no new dependency is needed for the core
reconstruction algorithm itself (only :mod:`miai_reconstruction.metrics`
adds a new dependency, scikit-image, for reconstruction-quality
metrics).

MIAI's datasets are NIfTI/DICOM images, not raw scanner k-space, so
:func:`simulate_kspace` derives k-space from an existing image via the
forward Fourier transform -- the same "simulate, then invert" pattern
:mod:`miai_diffusion` uses for its own from-scratch DDPM (there, noise
is added synthetically because no paired noisy/clean dataset exists
either).
"""

from __future__ import annotations

from typing import Literal

import torch

from miai_core.config import MIAIBaseConfig
from miai_reconstruction.exceptions import ReconstructionError


class KSpaceReconstructionConfig(MIAIBaseConfig):
    """Configuration for :func:`simulate_kspace` / :func:`reconstruct_from_kspace`.

    Attributes:
        norm: Normalization mode forwarded to :func:`torch.fft.fftn` /
            :func:`torch.fft.ifftn`. ``"ortho"`` (the default) makes
            the forward and inverse transforms exact inverses of each
            other with matching scaling, which is what makes the
            round trip ``reconstruct_from_kspace(simulate_kspace(x))
            == x`` hold (up to floating-point precision).
    """

    norm: Literal["ortho", "backward", "forward"] = "ortho"


def simulate_kspace(
    image: torch.Tensor, config: KSpaceReconstructionConfig | None = None
) -> torch.Tensor:
    """Compute the (simulated) k-space of ``image`` via the forward FFT.

    Args:
        image: A real-valued image or volume tensor.
        config: Reconstruction parameters. Uses defaults if ``None``.

    Returns:
        A complex-valued k-space tensor, the same shape as ``image``.
    """
    config = config or KSpaceReconstructionConfig()
    shifted = torch.fft.ifftshift(image)
    kspace = torch.fft.fftn(shifted, norm=config.norm)
    return torch.fft.fftshift(kspace)


def reconstruct_from_kspace(
    kspace: torch.Tensor, config: KSpaceReconstructionConfig | None = None
) -> torch.Tensor:
    """Reconstruct a (magnitude) image from k-space via the inverse FFT.

    Args:
        kspace: A complex-valued k-space tensor.
        config: Reconstruction parameters. Uses defaults if ``None``.
            Must match the ``config`` used to produce ``kspace`` (via
            :func:`simulate_kspace` or otherwise) for the two to be
            exact inverses.

    Returns:
        A real-valued (magnitude) image tensor, the same shape as
        ``kspace``.
    """
    config = config or KSpaceReconstructionConfig()
    shifted = torch.fft.ifftshift(kspace)
    image = torch.fft.ifftn(shifted, norm=config.norm)
    image = torch.fft.fftshift(image)
    return image.abs()


class UndersamplingConfig(MIAIBaseConfig):
    """Configuration for :func:`build_undersampling_mask`.

    Models the standard accelerated-MRI acquisition pattern: the
    central region of k-space (which carries most of the image's
    energy/contrast) is always fully sampled, while the remaining
    lines are randomly subsampled to hit a target acceleration factor
    -- the same design used by public MRI reconstruction benchmarks
    (e.g. fastMRI).

    Attributes:
        acceleration: Target undersampling factor. ``4.0`` means
            roughly 1/4 of the non-center lines are kept.
        center_fraction: Fraction of lines, centered on the k-space
            center, that are always fully sampled.
        axis: Which array axis is the phase-encode direction being
            undersampled (real MRI acquisitions fully sample the
            frequency-encode direction and undersample phase-encode).
        seed: Random seed, so the mask is reproducible.
    """

    acceleration: float = 4.0
    center_fraction: float = 0.08
    axis: int = -1
    seed: int = 0


def build_undersampling_mask(shape: tuple[int, ...], config: UndersamplingConfig) -> torch.Tensor:
    """Build a 1D-per-line undersampling mask, broadcastable over ``shape``.

    Args:
        shape: Shape of the k-space tensor the mask will be applied
            to.
        config: Undersampling parameters.

    Returns:
        A real-valued tensor of shape ``shape``, with each line along
        ``config.axis`` either all-ones (kept) or all-zeros (dropped).

    Raises:
        ReconstructionError: If ``config.acceleration`` is less than
            ``1.0`` (which would mean *more* than fully sampled).
    """
    if config.acceleration < 1.0:
        raise ReconstructionError(f"acceleration must be >= 1.0, got {config.acceleration}.")

    num_lines = shape[config.axis]
    num_center_lines = max(1, round(num_lines * config.center_fraction))
    center_start = (num_lines - num_center_lines) // 2

    line_mask = torch.zeros(num_lines)
    line_mask[center_start : center_start + num_center_lines] = 1.0

    num_remaining = num_lines - num_center_lines
    num_to_sample = max(0, round(num_remaining / config.acceleration))

    remaining_indices = torch.cat(
        [torch.arange(0, center_start), torch.arange(center_start + num_center_lines, num_lines)]
    )
    generator = torch.Generator().manual_seed(config.seed)
    if num_to_sample > 0 and len(remaining_indices) > 0:
        num_to_sample = min(num_to_sample, len(remaining_indices))
        perm = torch.randperm(len(remaining_indices), generator=generator)
        chosen = remaining_indices[perm[:num_to_sample]]
        line_mask[chosen] = 1.0

    view_shape = [1] * len(shape)
    view_shape[config.axis] = num_lines
    return line_mask.view(*view_shape).expand(*shape).clone()


def apply_undersampling(kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero out the k-space lines ``mask`` marks as unsampled."""
    return kspace * mask
