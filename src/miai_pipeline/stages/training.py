"""Training stage: trains a MIAI segmentation model via MONAI."""

from __future__ import annotations

from miai_core.config import MIAIBaseConfig
from miai_core.logging import get_logger
from miai_datasets.config import DataLoaderConfig
from miai_datasets.loaders import build_dataloader, build_dataset
from miai_datasets.manifest import manifest_split_to_data_dicts
from miai_pipeline.context import PipelineContext
from miai_pipeline.exceptions import StageError
from miai_pipeline.stage import PipelineStage
from miai_segmentation.models import UNetConfig, build_unet
from miai_segmentation.train import TrainingConfig, train_model
from miai_transforms.compose import build_transforms
from miai_transforms.config import TransformConfig

logger = get_logger(__name__)


class TrainingStageConfig(MIAIBaseConfig):
    """Configuration for :class:`TrainingStage`.

    Attributes:
        checkpoint_dir: Directory the trained model's checkpoint is
            written to.
        train_transforms: Transform pipeline applied to training cases
            (typically includes random augmentation).
        val_transforms: Transform pipeline applied to validation cases
            (typically the same deterministic steps as
            ``train_transforms``, without augmentation).
        unet: Model architecture configuration.
        training: Training hyperparameters.
        dataloader: Batching/loading configuration. ``shuffle`` is
            forced to ``True`` for the training split and ``False`` for
            the validation split, regardless of this value.
    """

    checkpoint_dir: str
    train_transforms: TransformConfig
    val_transforms: TransformConfig
    unet: UNetConfig = UNetConfig()
    training: TrainingConfig = TrainingConfig()
    dataloader: DataLoaderConfig = DataLoaderConfig()


class TrainingStage(PipelineStage):
    """Train a MONAI UNet on the dataset split produced by an earlier stage.

    Reads:
        ``manifest`` (``dict[str, list]``): the train/val/test split, as
        produced by
        :class:`~miai_pipeline.stages.dataset.DatasetStage`, built with
        a ``label_context_key`` so its entries are
        ``{"image": ..., "label": ...}`` mappings.

    Writes:
        ``model_checkpoint_path`` (``str``): path to the trained
        model's best-validation-Dice checkpoint.
    """

    name = "training"
    config_cls = TrainingStageConfig

    def __init__(self, config: TrainingStageConfig) -> None:
        """Store this stage's configuration."""
        self.config = config

    def run(self, context: PipelineContext) -> PipelineContext:
        """Run the stage; see the class docstring for its read/write contract."""
        manifest = context.require("manifest")
        train_entries = manifest.get("train", [])
        val_entries = manifest.get("val", [])
        if not train_entries:
            raise StageError("manifest['train'] is empty; nothing to train on.")

        train_dicts = manifest_split_to_data_dicts(train_entries)
        train_dataset = build_dataset(
            train_dicts,
            build_transforms(self.config.train_transforms),
            cache_rate=self.config.dataloader.cache_rate,
        )
        train_loader = build_dataloader(
            train_dataset, self.config.dataloader.model_copy(update={"shuffle": True})
        )

        val_loader = None
        if val_entries:
            val_dicts = manifest_split_to_data_dicts(val_entries)
            val_dataset = build_dataset(
                val_dicts,
                build_transforms(self.config.val_transforms),
                cache_rate=self.config.dataloader.cache_rate,
            )
            val_loader = build_dataloader(
                val_dataset, self.config.dataloader.model_copy(update={"shuffle": False})
            )

        model = build_unet(self.config.unet)
        checkpoint_path = train_model(
            model, train_loader, val_loader, self.config.training, self.config.checkpoint_dir
        )

        context.set("model_checkpoint_path", str(checkpoint_path))
        return context
