"""Denoising stage: runs reverse diffusion over a list of volumes."""

from __future__ import annotations

from miai_core.config import MIAIBaseConfig
from miai_core.logging import get_logger
from miai_diffusion.denoise import DenoiseConfig, run_denoising
from miai_diffusion.model import DiffusionUNetConfig, build_diffusion_unet
from miai_diffusion.schedule import NoiseSchedule, NoiseScheduleConfig
from miai_pipeline.context import PipelineContext
from miai_pipeline.exceptions import StageError
from miai_pipeline.stage import PipelineStage

logger = get_logger(__name__)


class DenoisingStageConfig(MIAIBaseConfig):
    """Configuration for :class:`DenoisingStage`.

    Attributes:
        output_dir: Directory denoised volumes are written to.
        unet: Model architecture configuration. Must match the
            architecture the checkpoint was trained with.
        schedule: Noise schedule configuration. Must match the schedule
            the checkpoint was trained under.
        denoise: Reverse-diffusion parameters.
        context_key: Which context key holds the list of case paths to
            denoise -- typically ``"preprocessed_paths"``.
        checkpoint_path: Path to a trained checkpoint. If ``None``
            (default), falls back to the ``diffusion_checkpoint_path``
            written by an earlier
            :class:`~miai_pipeline.stages.diffusion_training.DiffusionTrainingStage`
            in the same pipeline run.
    """

    output_dir: str
    unet: DiffusionUNetConfig = DiffusionUNetConfig()
    schedule: NoiseScheduleConfig = NoiseScheduleConfig()
    denoise: DenoiseConfig = DenoiseConfig()
    context_key: str = "preprocessed_paths"
    checkpoint_path: str | None = None


class DenoisingStage(PipelineStage):
    """Denoise a list of volumes with a trained diffusion model.

    Reads:
        ``<config.context_key>`` (``list[Path]``, default
        ``"preprocessed_paths"``): the volumes to denoise.
        ``diffusion_checkpoint_path`` (``str``, unless
        ``config.checkpoint_path`` is set): the trained model, as
        produced by
        :class:`~miai_pipeline.stages.diffusion_training.DiffusionTrainingStage`.

    Writes:
        ``denoised_paths`` (``list[Path]``): one denoised volume per
        input case.
    """

    name = "denoising"
    config_cls = DenoisingStageConfig

    def __init__(self, config: DenoisingStageConfig) -> None:
        """Store this stage's configuration."""
        self.config = config

    def run(self, context: PipelineContext) -> PipelineContext:
        """Run the stage; see the class docstring for its read/write contract."""
        source_paths = context.require(self.config.context_key)
        if not source_paths:
            raise StageError(f"'{self.config.context_key}' is empty; nothing to denoise.")

        checkpoint_path = self.config.checkpoint_path or context.require(
            "diffusion_checkpoint_path"
        )

        model = build_diffusion_unet(self.config.unet)
        schedule = NoiseSchedule(self.config.schedule, device=self.config.denoise.device)

        denoised_paths = run_denoising(
            model,
            schedule,
            checkpoint_path,
            [str(p) for p in source_paths],
            self.config.denoise,
            self.config.output_dir,
        )

        context.set("denoised_paths", denoised_paths)
        return context
