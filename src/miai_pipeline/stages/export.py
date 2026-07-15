"""Export stage: bundles a trained model as a portable artifact."""

from __future__ import annotations

from miai_core.config import MIAIBaseConfig
from miai_core.logging import get_logger
from miai_deploy.bundle import BundleMetadata, write_bundle
from miai_deploy.export import ExportConfig
from miai_pipeline.context import PipelineContext
from miai_pipeline.stage import PipelineStage
from miai_segmentation.models import UNetConfig, build_unet

logger = get_logger(__name__)


class ExportStageConfig(MIAIBaseConfig):
    """Configuration for :class:`ExportStage`.

    Attributes:
        output_dir: Directory the deployment bundle is written to.
        unet: Model architecture configuration. Must match the
            architecture the checkpoint was trained with.
        export: Export format/tracing parameters.
        metadata: Reproducibility metadata for this bundle.
        checkpoint_path: Path to a trained checkpoint. If ``None``
            (default), falls back to the ``model_checkpoint_path``
            written by an earlier
            :class:`~miai_pipeline.stages.training.TrainingStage` in the
            same pipeline run.
    """

    output_dir: str
    unet: UNetConfig = UNetConfig()
    export: ExportConfig = ExportConfig()
    metadata: BundleMetadata
    checkpoint_path: str | None = None


class ExportStage(PipelineStage):
    """Export a trained model and write it as a deployment bundle.

    Reads:
        ``model_checkpoint_path`` (``str``, unless
        ``config.checkpoint_path`` is set): the trained model, as
        produced by
        :class:`~miai_pipeline.stages.training.TrainingStage`.

    Writes:
        ``deploy_bundle_path`` (``Path``): the bundle directory,
        containing the exported model and ``metadata.yaml``.
    """

    name = "export"
    config_cls = ExportStageConfig

    def __init__(self, config: ExportStageConfig) -> None:
        self.config = config

    def run(self, context: PipelineContext) -> PipelineContext:
        checkpoint_path = self.config.checkpoint_path or context.require("model_checkpoint_path")

        model = build_unet(self.config.unet)
        bundle_path = write_bundle(
            model,
            checkpoint_path,
            self.config.export,
            self.config.metadata,
            self.config.output_dir,
        )

        context.set("deploy_bundle_path", bundle_path)
        return context
