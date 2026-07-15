"""Reconstruction stage: simulates and reconstructs k-space for a list of volumes."""

from __future__ import annotations

from miai_core.config import MIAIBaseConfig
from miai_core.logging import get_logger
from miai_pipeline.context import PipelineContext
from miai_pipeline.exceptions import StageError
from miai_pipeline.stage import PipelineStage
from miai_reconstruction.kspace import KSpaceReconstructionConfig, UndersamplingConfig
from miai_reconstruction.run import run_kspace_reconstruction

logger = get_logger(__name__)


class ReconstructionStageConfig(MIAIBaseConfig):
    """Configuration for :class:`ReconstructionStage`.

    Attributes:
        output_dir: Directory reconstructed volumes are written to.
        reconstruction: FFT normalization parameters.
        undersampling: If set, simulates an accelerated (undersampled)
            acquisition before reconstructing; if ``None`` (default),
            reconstructs from the full simulated k-space.
        context_key: Which context key holds the list of case paths to
            reconstruct -- typically ``"preprocessed_paths"``.
    """

    output_dir: str
    reconstruction: KSpaceReconstructionConfig = KSpaceReconstructionConfig()
    undersampling: UndersamplingConfig | None = None
    context_key: str = "preprocessed_paths"


class ReconstructionStage(PipelineStage):
    """Simulate k-space for a list of volumes and reconstruct them.

    Reads:
        ``<config.context_key>`` (``list[Path]``, default
        ``"preprocessed_paths"``): the volumes to simulate k-space for
        and reconstruct.

    Writes:
        ``reconstructed_paths`` (``list[Path]``): one reconstructed
        volume per input case.
    """

    name = "reconstruction"
    config_cls = ReconstructionStageConfig

    def __init__(self, config: ReconstructionStageConfig) -> None:
        self.config = config

    def run(self, context: PipelineContext) -> PipelineContext:
        source_paths = context.require(self.config.context_key)
        if not source_paths:
            raise StageError(f"'{self.config.context_key}' is empty; nothing to reconstruct.")

        reconstructed_paths = run_kspace_reconstruction(
            [str(p) for p in source_paths],
            self.config.reconstruction,
            self.config.undersampling,
            self.config.output_dir,
        )

        context.set("reconstructed_paths", reconstructed_paths)
        return context
