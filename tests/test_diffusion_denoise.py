"""Tests for miai_diffusion.denoise (tiny real tensors, CPU only)."""

from pathlib import Path

import pytest
import SimpleITK as sitk
import torch

from miai_diffusion.denoise import DenoiseConfig, denoise_volume, run_denoising
from miai_diffusion.exceptions import DiffusionError
from miai_diffusion.model import DiffusionUNetConfig, build_diffusion_unet
from miai_diffusion.schedule import NoiseSchedule, NoiseScheduleConfig

_UNET_CONFIG = DiffusionUNetConfig(
    in_channels=1, base_channels=4, channel_multipliers=(1, 2), time_embedding_dim=16
)


def test_denoise_volume_output_shape_matches_input() -> None:
    model = build_diffusion_unet(_UNET_CONFIG)
    schedule = NoiseSchedule(NoiseScheduleConfig(num_timesteps=20), device="cpu")
    config = DenoiseConfig(start_timestep=5, device="cpu")
    noisy = torch.randn(1, 1, 8, 8, 8)

    denoised = denoise_volume(model, schedule, noisy, config)

    assert denoised.shape == noisy.shape


def test_denoise_volume_out_of_range_start_timestep_raises() -> None:
    model = build_diffusion_unet(_UNET_CONFIG)
    schedule = NoiseSchedule(NoiseScheduleConfig(num_timesteps=20), device="cpu")
    config = DenoiseConfig(start_timestep=20, device="cpu")
    noisy = torch.randn(1, 1, 8, 8, 8)

    with pytest.raises(DiffusionError):
        denoise_volume(model, schedule, noisy, config)


@pytest.mark.slow
def test_run_denoising_writes_output_matching_reference_geometry(tmp_path: Path) -> None:
    size = 8
    arr = torch.randn(size, size, size).numpy().astype("float32")
    image = sitk.GetImageFromArray(arr)
    image_path = tmp_path / "noisy.nii.gz"
    sitk.WriteImage(image, str(image_path))

    model = build_diffusion_unet(_UNET_CONFIG)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    schedule = NoiseSchedule(NoiseScheduleConfig(num_timesteps=20), device="cpu")
    config = DenoiseConfig(start_timestep=5, device="cpu")

    fresh_model = build_diffusion_unet(_UNET_CONFIG)
    denoised_paths = run_denoising(
        fresh_model,
        schedule,
        str(checkpoint_path),
        [str(image_path)],
        config,
        str(tmp_path / "denoised"),
    )

    assert len(denoised_paths) == 1
    assert denoised_paths[0].exists()

    reference_image = sitk.ReadImage(str(image_path))
    denoised_image = sitk.ReadImage(str(denoised_paths[0]))
    assert denoised_image.GetSize() == reference_image.GetSize()
    assert denoised_image.GetSpacing() == reference_image.GetSpacing()
