"""MIAI Diffusion: a from-scratch DDPM for volume denoising, on PyTorch.

Implements the standard Denoising Diffusion Probabilistic Model (Ho,
Jain & Abbeel, 2020) end to end -- noise schedule
(:mod:`miai_diffusion.schedule`), a compact time-conditioned 3D UNet
(:mod:`miai_diffusion.model`), a training loop
(:mod:`miai_diffusion.train`), and reverse-diffusion denoising of real
noisy volumes (:mod:`miai_diffusion.denoise`) -- without a dependency on
MONAI's generative extension, matching MIAI's existing PyTorch +
SimpleITK stack. Used by
:class:`~miai_pipeline.stages.diffusion_training.DiffusionTrainingStage`
and :class:`~miai_pipeline.stages.denoising.DenoisingStage`.
"""

from miai_diffusion.denoise import DenoiseConfig, denoise_volume, run_denoising
from miai_diffusion.exceptions import DiffusionError
from miai_diffusion.model import DiffusionUNet, DiffusionUNetConfig, build_diffusion_unet
from miai_diffusion.schedule import NoiseSchedule, NoiseScheduleConfig
from miai_diffusion.train import DiffusionTrainingConfig, train_diffusion_model

__version__ = "0.1.0"

__all__ = [
    "NoiseSchedule",
    "NoiseScheduleConfig",
    "DiffusionUNet",
    "DiffusionUNetConfig",
    "build_diffusion_unet",
    "DiffusionTrainingConfig",
    "train_diffusion_model",
    "DenoiseConfig",
    "denoise_volume",
    "run_denoising",
    "DiffusionError",
    "__version__",
]
