"""Denoising a real (already-noisy) volume via reverse diffusion."""

from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk
import torch

from miai_core.config import MIAIBaseConfig
from miai_core.io import ensure_dir
from miai_core.logging import get_logger
from miai_diffusion.exceptions import DiffusionError
from miai_diffusion.schedule import NoiseSchedule

logger = get_logger(__name__)


class DenoiseConfig(MIAIBaseConfig):
    """Configuration for :func:`denoise_volume` / :func:`run_denoising`.

    Attributes:
        start_timestep: Which diffusion timestep to treat the noisy
            input as. Reverse diffusion runs from here down to 0.
            Smaller values assume the input is only lightly noisy
            (fewer, more conservative denoising steps, closer to the
            input); larger values assume heavier noise (more steps,
            more aggressive denoising, and more risk of altering real
            structure). This is the standard SDEdit-style trick for
            using a generative diffusion model as a restoration prior,
            rather than only for unconditional sampling from pure
            noise.
        device: ``"cpu"`` or ``"cuda"``.
    """

    start_timestep: int = 250
    device: str = "cpu"


def denoise_volume(
    model: torch.nn.Module, schedule: NoiseSchedule, noisy: torch.Tensor, config: DenoiseConfig
) -> torch.Tensor:
    """Denoise a batch of volumes by running the reverse diffusion process.

    Treats ``noisy`` as the schedule's ``x_t`` at
    ``config.start_timestep``, then repeatedly applies
    :meth:`~miai_diffusion.schedule.NoiseSchedule.p_sample_step` down to
    ``t=0``.

    Args:
        model: A trained noise-prediction model (e.g. from
            :func:`miai_diffusion.model.build_diffusion_unet`).
        schedule: The noise schedule ``model`` was trained under.
        noisy: The volume(s) to denoise, shape ``(B, C, ...)``.
        config: Denoising parameters.

    Returns:
        The denoised volume(s), same shape as ``noisy``.

    Raises:
        DiffusionError: If ``config.start_timestep`` is outside
            ``[0, schedule.config.num_timesteps)``.
    """
    if not 0 <= config.start_timestep < schedule.config.num_timesteps:
        raise DiffusionError(
            f"start_timestep={config.start_timestep} is outside the schedule's range "
            f"[0, {schedule.config.num_timesteps})."
        )

    device = torch.device(config.device)
    model = model.to(device)
    model.eval()

    x_t = noisy.to(device)
    for t in reversed(range(config.start_timestep + 1)):
        x_t = schedule.p_sample_step(model, x_t, t)
    return x_t


def run_denoising(
    model: torch.nn.Module,
    schedule: NoiseSchedule,
    checkpoint_path: str,
    source_paths: list[str],
    config: DenoiseConfig,
    output_dir: str,
) -> list[Path]:
    """Denoise a list of volumes on disk and write the results as NIfTI.

    Reads each source volume via SimpleITK, denoises it with
    :func:`denoise_volume`, and writes the result copying the source
    file's spatial metadata (spacing/origin/direction) -- the same
    SimpleITK-only I/O convention used throughout MIAI (see
    :func:`miai_segmentation.infer.run_inference`).

    Args:
        model: An untrained model with the same architecture used to
            produce ``checkpoint_path``.
        schedule: The noise schedule ``model`` was trained under.
        checkpoint_path: Path to a state dict saved by
            :func:`miai_diffusion.train.train_diffusion_model`.
        source_paths: Volumes to denoise.
        config: Denoising parameters.
        output_dir: Directory denoised volumes are written to (created
            if missing).

    Returns:
        One denoised file path per entry in ``source_paths``, in order.
    """
    device = torch.device(config.device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    out_dir = ensure_dir(output_dir)
    denoised_paths: list[Path] = []

    for source_path in source_paths:
        reference_image = sitk.ReadImage(str(source_path))
        array = sitk.GetArrayFromImage(reference_image).astype("float32")
        noisy = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)

        denoised = denoise_volume(model, schedule, noisy, config)
        denoised_array = denoised.squeeze(0).squeeze(0).cpu().numpy()

        denoised_image = sitk.GetImageFromArray(denoised_array)
        denoised_image.CopyInformation(reference_image)

        stem = Path(source_path).name.removesuffix(".nii.gz").removesuffix(".nii")
        out_path = out_dir / f"{stem}_denoised.nii.gz"
        sitk.WriteImage(denoised_image, str(out_path))
        denoised_paths.append(out_path)
        logger.info("Wrote denoised volume for %s to %s", source_path, out_path)

    return denoised_paths
