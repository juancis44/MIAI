"""End-to-end example: DDPM training and denoising, an optional MIAI stage.

Demonstrates :mod:`miai_diffusion` directly (not via a pipeline YAML
config, unlike ``examples/segmentation_pipeline.py``), since diffusion
training/denoising is an optional stage outside the main segmentation
workflow -- see the six optional stages listed in
``miai_pipeline``'s package docstring.

Trains a compact 3D DDPM (Ho, Jain & Abbeel, 2020) on a handful of
synthetic volumes, then simulates a "real noisy scan" by corrupting a
known-clean volume with real forward-diffusion noise
(:meth:`~miai_diffusion.schedule.NoiseSchedule.q_sample`) and uses the
trained model to denoise it via reverse diffusion
(:func:`~miai_diffusion.denoise.run_denoising`), reporting whether the
denoised result is closer to the original clean volume than the noisy
input was.

No real dataset or GPU is required; this runs on CPU in well under a
minute.

Run:
    python examples/diffusion_denoising.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
import torch

from miai_core.logging import configure_logging, get_logger
from miai_diffusion import (
    DenoiseConfig,
    DiffusionTrainingConfig,
    DiffusionUNetConfig,
    NoiseSchedule,
    NoiseScheduleConfig,
    build_diffusion_unet,
    run_denoising,
    train_diffusion_model,
)

logger = get_logger(__name__)

EXAMPLE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXAMPLE_DIR / "output" / "diffusion"

#: (depth, rows, columns). Small and divisible by the UNet's one
#: downsampling step (channel_multipliers below has 2 levels).
VOLUME_SHAPE = (16, 16, 16)
NUM_TRAINING_VOLUMES = 8
NUM_TIMESTEPS = 100
#: How noisy the simulated "scan" is treated as -- must be < NUM_TIMESTEPS.
#: Mid-range: noisy enough that denoising has real work to do, without
#: destroying the signal so completely that reverse diffusion can't
#: recover anything resembling the original in this tiny demo.
START_TIMESTEP = 60

_UNET_CONFIG = DiffusionUNetConfig(
    in_channels=1, base_channels=8, channel_multipliers=(1, 2), time_embedding_dim=16
)


def _make_clean_volume(rng: np.random.Generator) -> npt.NDArray[np.float32]:
    """A simple synthetic volume: a centered soft spherical blob.

    Stands in for a real anatomical structure -- smooth and centered,
    but with a randomized radius per call, so the training set isn't
    ``NUM_TRAINING_VOLUMES`` copies of the exact same array.
    """
    depth, rows, columns = VOLUME_SHAPE
    zz, yy, xx = np.meshgrid(
        np.linspace(-1, 1, depth),
        np.linspace(-1, 1, rows),
        np.linspace(-1, 1, columns),
        indexing="ij",
    )
    radius = rng.uniform(0.4, 0.6)
    blob = np.clip(1.0 - (zz**2 + yy**2 + xx**2) / radius**2, 0.0, 1.0)
    return blob.astype(np.float32)


def _make_training_batches(rng: np.random.Generator) -> list[dict[str, torch.Tensor]]:
    """Batches of one volume each, the shape :func:`train_diffusion_model` expects."""
    return [
        {"image": torch.from_numpy(_make_clean_volume(rng)).unsqueeze(0).unsqueeze(0)}
        for _ in range(NUM_TRAINING_VOLUMES)
    ]


def _write_volume(path: Path, volume: npt.NDArray[np.float32]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(path))


def main() -> None:
    configure_logging(level="INFO", force=True)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    rng = np.random.default_rng(0)
    schedule = NoiseSchedule(NoiseScheduleConfig(num_timesteps=NUM_TIMESTEPS), device="cpu")

    model = build_diffusion_unet(_UNET_CONFIG)
    training_config = DiffusionTrainingConfig(max_epochs=40, learning_rate=1e-3, device="cpu")
    checkpoint_path = train_diffusion_model(
        model,
        _make_training_batches(rng),
        schedule,
        training_config,
        str(OUTPUT_DIR / "checkpoints"),
    )
    logger.info("Trained diffusion model checkpoint: %s", checkpoint_path)

    # Simulate a "real noisy scan": corrupt one known-clean volume with
    # real forward-diffusion noise at START_TIMESTEP, then check whether
    # reverse diffusion recovers something closer to the original than
    # the noisy input was.
    clean = _make_clean_volume(rng)
    _write_volume(OUTPUT_DIR / "clean_volume.nii.gz", clean)

    clean_tensor = torch.from_numpy(clean).unsqueeze(0).unsqueeze(0)
    noise = torch.randn_like(clean_tensor)
    timestep = torch.tensor([START_TIMESTEP])
    noisy_tensor = schedule.q_sample(clean_tensor, timestep, noise)
    noisy_path = OUTPUT_DIR / "noisy_volume.nii.gz"
    _write_volume(noisy_path, noisy_tensor.squeeze(0).squeeze(0).numpy())

    denoise_config = DenoiseConfig(start_timestep=START_TIMESTEP, device="cpu")
    denoised_paths = run_denoising(
        build_diffusion_unet(_UNET_CONFIG),
        schedule,
        str(checkpoint_path),
        [str(noisy_path)],
        denoise_config,
        str(OUTPUT_DIR / "denoised"),
    )

    noisy_array = sitk.GetArrayFromImage(sitk.ReadImage(str(noisy_path)))
    denoised_array = sitk.GetArrayFromImage(sitk.ReadImage(str(denoised_paths[0])))
    noisy_mse = float(np.mean((noisy_array - clean) ** 2))
    denoised_mse = float(np.mean((denoised_array - clean) ** 2))

    print()
    print("=== Diffusion denoising finished ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Noisy vs. clean MSE:    {noisy_mse:.4f}")
    print(f"Denoised vs. clean MSE: {denoised_mse:.4f}  (lower is better)")
    print(f"Volumes written under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
