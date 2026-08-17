"""Visualization stage: writes a QC slice montage per case."""

from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk

from miai_core.config import MIAIBaseConfig
from miai_core.io import ensure_dir
from miai_core.logging import get_logger
from miai_pipeline.context import PipelineContext
from miai_pipeline.exceptions import StageError
from miai_pipeline.stage import PipelineStage
from miai_visualization.slices import PlotMontageConfig, plot_montage

logger = get_logger(__name__)


class VisualizationStageConfig(MIAIBaseConfig):
    """Configuration for :class:`VisualizationStage`.

    Attributes:
        output_dir: Directory QC montage PNGs are written to.
        montage: Montage plotting parameters.
        context_key: Which context key holds the list of case paths to
            visualize -- typically ``"preprocessed_paths"``.
    """

    output_dir: str
    montage: PlotMontageConfig = PlotMontageConfig()
    context_key: str = "preprocessed_paths"


class VisualizationStage(PipelineStage):
    """Write a slice-montage PNG for each case in a run, for quick QC.

    Reads:
        ``<config.context_key>`` (``list[Path]``, default
        ``"preprocessed_paths"``): the volumes to visualize.

    Writes:
        ``qc_visualization_paths`` (``list[Path]``): one montage PNG
        per input case.
    """

    name = "visualization"
    config_cls = VisualizationStageConfig

    def __init__(self, config: VisualizationStageConfig) -> None:
        """Store this stage's configuration."""
        self.config = config

    def run(self, context: PipelineContext) -> PipelineContext:
        """Run the stage; see the class docstring for its read/write contract."""
        source_paths = context.require(self.config.context_key)
        if not source_paths:
            raise StageError(f"'{self.config.context_key}' is empty; nothing to visualize.")

        out_dir = ensure_dir(self.config.output_dir)
        qc_paths: list[Path] = []

        for source_path in source_paths:
            image = sitk.ReadImage(str(source_path))
            array = sitk.GetArrayFromImage(image)

            stem = Path(str(source_path)).name.removesuffix(".nii.gz").removesuffix(".nii")
            out_path = out_dir / f"{stem}_qc.png"
            plot_montage(array, str(out_path), self.config.montage)
            qc_paths.append(out_path)
            logger.info("Wrote QC montage for %s to %s", source_path, out_path)

        context.set("qc_visualization_paths", qc_paths)
        return context
