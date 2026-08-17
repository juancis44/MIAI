"""Diffusion training stage: trains a DDPM noise-prediction model."""

from __future__ import annotations

from miai_core.config import MIAIBaseConfig
from miai_core.logging import get_logger
from miai_datasets.config import DataLoaderConfig
from miai_datasets.loaders import build_dataloader, build_dataset
from miai_datasets.manifest import manifest_split_to_data_dicts
from miai_diffusion.model import DiffusionUNetConfig, build_diffusion_unet
from miai_diffusion.schedule import NoiseSchedule, NoiseScheduleConfig
from miai_diffusion.train import DiffusionTrainingConfig, train_diffusion_model
from miai_pipeline.context import PipelineContext
from miai_pipeline.exceptions import StageError
from miai_pipeline.stage import PipelineStage
from miai_transforms.compose import build_transforms
from miai_transforms.config import TransformConfig

logger = get_logger(__name__)


class DiffusionTrainingStageConfig(MIAIBaseConfig):
    """Configuration for :class:`DiffusionTrainingStage`.

    Attributes:
        checkpoint_dir: Directory the trained model's checkpoint is
            written to.
        transforms: Transform pipeline applied to training cases.
        unet: Model architecture configuration.
        schedule: Noise schedule configuration.
        training: Training hyperparameters.
        dataloader: Batching/loading configuration.
    """

    checkpoint_dir: str
    transforms: TransformConfig
    unet: DiffusionUNetConfig = DiffusionUNetConfig()
    schedule: NoiseScheduleConfig = NoiseScheduleConfig()
    training: DiffusionTrainingConfig = DiffusionTrainingConfig()
    dataloader: DataLoaderConfig = DataLoaderConfig()


class DiffusionTrainingStage(PipelineStage):
    """Train a DDPM noise-prediction model, unconditionally, on the training split.

    Reads:
        ``manifest`` (``dict[str, list]``): only ``manifest["train"]``
        is used -- diffusion training here is unconditional (no
        labels), unlike
        :class:`~miai_pipeline.stages.training.TrainingStage`.

    Writes:
        ``diffusion_checkpoint_path`` (``str``): path to the trained
        model's checkpoint.
    """

    name = "diffusion_training"
    config_cls = DiffusionTrainingStageConfig

    def __init__(self, config: DiffusionTrainingStageConfig) -> None:
        """Store this stage's configuration."""
        self.config = config

    def run(self, context: PipelineContext) -> PipelineContext:
        """Run the stage; see the class docstring for its read/write contract."""
        manifest = context.require("manifest")
        train_entries = manifest.get("train", [])
        if not train_entries:
            raise StageError("manifest['train'] is empty; nothing to train on.")

        train_dicts = manifest_split_to_data_dicts(train_entries)
        train_dataset = build_dataset(
            train_dicts,
            build_transforms(self.config.transforms),
            cache_rate=self.config.dataloader.cache_rate,
        )
        train_loader = build_dataloader(
            train_dataset, self.config.dataloader.model_copy(update={"shuffle": True})
        )

        model = build_diffusion_unet(self.config.unet)
        schedule = NoiseSchedule(self.config.schedule, device=self.config.training.device)

        checkpoint_path = train_diffusion_model(
            model, train_loader, schedule, self.config.training, self.config.checkpoint_dir
        )

        context.set("diffusion_checkpoint_path", str(checkpoint_path))
        return context
