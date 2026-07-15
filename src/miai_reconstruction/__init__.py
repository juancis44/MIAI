"""miai-reconstruction: MRI k-space reconstruction.

MIAI's reference reconstruction task is MRI reconstruction from
k-space via the (inverse) Fourier transform, implemented on
:mod:`torch.fft` -- see :mod:`miai_reconstruction.kspace` for the core
algorithm (including zero-filled undersampled reconstruction), and
:mod:`miai_reconstruction.run` for applying it to a list of volumes on
disk. :mod:`miai_reconstruction.metrics` adds PSNR/SSIM
reconstruction-quality metrics via scikit-image.
"""

from __future__ import annotations

from miai_reconstruction.exceptions import ReconstructionError
from miai_reconstruction.kspace import (
    KSpaceReconstructionConfig,
    UndersamplingConfig,
    apply_undersampling,
    build_undersampling_mask,
    reconstruct_from_kspace,
    simulate_kspace,
)
from miai_reconstruction.metrics import reconstruction_quality
from miai_reconstruction.run import run_kspace_reconstruction

__version__ = "0.1.0"

__all__ = [
    "ReconstructionError",
    "KSpaceReconstructionConfig",
    "UndersamplingConfig",
    "simulate_kspace",
    "reconstruct_from_kspace",
    "build_undersampling_mask",
    "apply_undersampling",
    "reconstruction_quality",
    "run_kspace_reconstruction",
]
