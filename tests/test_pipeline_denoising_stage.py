"""Integration test for the concrete DenoisingStage."""

from pathlib import Path

import pytest
import torch

from conftest import make_offset_cube_volume
from miai_diffusion.denoise import DenoiseConfig
from miai_diffusion.model import DiffusionUNetConfig, build_diffusion_unet
from miai_diffusion.schedule import NoiseScheduleConfig
from miai_pipeline.context import PipelineContext
from miai_pipeline.exceptions import StageError
from miai_pipeline.stages.denoising import DenoisingStage, DenoisingStageConfig

_UNET_CONFIG = DiffusionUNetConfig(
    in_channels=1, base_channels=4, channel_multipliers=(1, 2), time_embedding_dim=16
)


@pytest.mark.slow
def test_denoising_stage_writes_denoised_output_using_context_checkpoint(tmp_path: Path) -> None:
    image_path = make_offset_cube_volume(tmp_path / "data", name="case0", size=(8, 8, 8))
    checkpoint_path = tmp_path / "model.pt"
    torch.save(build_diffusion_unet(_UNET_CONFIG).state_dict(), checkpoint_path)

    ctx = PipelineContext()
    ctx.set("preprocessed_paths", [image_path])
    ctx.set("diffusion_checkpoint_path", str(checkpoint_path))

    stage = DenoisingStage(
        DenoisingStageConfig(
            output_dir=str(tmp_path / "denoised"),
            unet=_UNET_CONFIG,
            schedule=NoiseScheduleConfig(num_timesteps=20),
            denoise=DenoiseConfig(start_timestep=5, device="cpu"),
        )
    )

    result = stage.run(ctx)

    denoised_paths = result.require("denoised_paths")
    assert len(denoised_paths) == 1
    assert Path(denoised_paths[0]).exists()


def test_denoising_stage_empty_context_key_raises(tmp_path: Path) -> None:
    ctx = PipelineContext()
    ctx.set("preprocessed_paths", [])

    stage = DenoisingStage(DenoisingStageConfig(output_dir=str(tmp_path / "denoised")))

    with pytest.raises(StageError):
        stage.run(ctx)
