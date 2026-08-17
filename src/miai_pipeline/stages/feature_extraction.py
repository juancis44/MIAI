"""Feature-extraction stage: embeds a list of volumes with a pretrained model."""

from __future__ import annotations

from miai_core.config import MIAIBaseConfig
from miai_core.logging import get_logger
from miai_foundation_models.extractor import FeatureExtractor, FeatureExtractorConfig
from miai_foundation_models.run import extract_embeddings_for_paths
from miai_pipeline.context import PipelineContext
from miai_pipeline.exceptions import StageError
from miai_pipeline.stage import PipelineStage

logger = get_logger(__name__)


class FeatureExtractionStageConfig(MIAIBaseConfig):
    """Configuration for :class:`FeatureExtractionStage`.

    Attributes:
        output_dir: Directory embeddings are written to.
        extractor: Which pretrained model to use and how to pool its
            output into a per-volume embedding.
        context_key: Which context key holds the list of case paths to
            embed -- typically ``"preprocessed_paths"``.
    """

    output_dir: str
    extractor: FeatureExtractorConfig = FeatureExtractorConfig()
    context_key: str = "preprocessed_paths"


class FeatureExtractionStage(PipelineStage):
    """Extract a pretrained-model embedding for each volume in a run.

    Reads:
        ``<config.context_key>`` (``list[Path]``, default
        ``"preprocessed_paths"``): the volumes to embed.

    Writes:
        ``embedding_paths`` (``list[Path]``): one embedding file per
        input case.
    """

    name = "feature_extraction"
    config_cls = FeatureExtractionStageConfig

    def __init__(self, config: FeatureExtractionStageConfig) -> None:
        """Store this stage's configuration."""
        self.config = config

    def run(self, context: PipelineContext) -> PipelineContext:
        """Run the stage; see the class docstring for its read/write contract."""
        source_paths = context.require(self.config.context_key)
        if not source_paths:
            raise StageError(f"'{self.config.context_key}' is empty; nothing to embed.")

        extractor = FeatureExtractor.from_pretrained(self.config.extractor)

        embedding_paths = extract_embeddings_for_paths(
            extractor,
            [str(p) for p in source_paths],
            self.config.output_dir,
        )

        context.set("embedding_paths", embedding_paths)
        return context
