"""Inference stage: runs sliding-window inference via MONAI."""

from __future__ import annotations

from miai_core.config import MIAIBaseConfig
from miai_core.logging import get_logger
from miai_datasets.config import DataLoaderConfig
from miai_datasets.loaders import build_dataloader, build_dataset
from miai_datasets.manifest import manifest_split_to_data_dicts
from miai_datasets.slices import expand_to_slice_dicts
from miai_pipeline.context import PipelineContext
from miai_pipeline.exceptions import StageError
from miai_pipeline.stage import PipelineStage
from miai_segmentation.modality import (
    SegmentationInferenceConfig,
    SegmentationModalityConfig,
    build_model_for_modality,
)
from miai_segmentation.three_d.infer import run_inference as run_three_d_inference
from miai_segmentation.two_d.infer import run_case_inference
from miai_transforms.compose import build_transforms
from miai_transforms.config import TransformConfig

logger = get_logger(__name__)


class InferenceStageConfig(MIAIBaseConfig):
    """Configuration for :class:`InferenceStage`.

    Attributes:
        output_dir: Directory predictions are written to.
        transforms: Transform pipeline applied to test cases (should
            match the training stage's ``val_transforms`` -- no random
            augmentation).
        architecture: Segmentation modality and architecture selection
            (see :class:`~miai_segmentation.modality.
            SegmentationModalityConfig`). Must match the modality and
            architecture the checkpoint was trained with. When
            ``architecture.modality`` is ``"two_d"`` or ``"two_half_d"``,
            each test case is expanded into one dict per slice before
            ``transforms`` runs (see
            :class:`~miai_pipeline.stages.training.TrainingStageConfig`'s
            ``architecture`` docstring), and predictions are reassembled
            into one volume per case.
        inference: Sliding-window inference parameters, keyed by
            modality (see :class:`~miai_segmentation.modality.
            SegmentationInferenceConfig`).
        checkpoint_path: Path to a trained checkpoint. If ``None``
            (default), falls back to the ``model_checkpoint_path``
            written by an earlier
            :class:`~miai_pipeline.stages.training.TrainingStage` in the
            same pipeline run.
    """

    output_dir: str
    transforms: TransformConfig
    architecture: SegmentationModalityConfig = SegmentationModalityConfig()
    inference: SegmentationInferenceConfig = SegmentationInferenceConfig()
    checkpoint_path: str | None = None


class InferenceStage(PipelineStage):
    """Run a trained model over the test split to produce predictions.

    Reads:
        ``model_checkpoint_path`` (``str``, unless
        ``config.checkpoint_path`` is set): the trained model, as
        produced by
        :class:`~miai_pipeline.stages.training.TrainingStage`.
        ``manifest`` (``dict[str, list]``): provides the ``test`` cases
        to run inference on.

    Writes:
        ``prediction_paths`` (``list[Path]``): one prediction volume
        per test case.
    """

    name = "inference"
    config_cls = InferenceStageConfig

    def __init__(self, config: InferenceStageConfig) -> None:
        """Store this stage's configuration."""
        self.config = config

    def run(self, context: PipelineContext) -> PipelineContext:
        """Run the stage; see the class docstring for its read/write contract."""
        manifest = context.require("manifest")
        test_entries = manifest.get("test", [])
        if not test_entries:
            raise StageError("manifest['test'] is empty; nothing to run inference on.")

        test_dicts = manifest_split_to_data_dicts(test_entries)
        source_paths = [d["image"] for d in test_dicts]

        modality = self.config.architecture.modality
        checkpoint_path = self.config.checkpoint_path or context.require("model_checkpoint_path")
        model = build_model_for_modality(self.config.architecture)

        if modality == "three_d":
            test_dataset = build_dataset(test_dicts, build_transforms(self.config.transforms))
            test_loader = build_dataloader(
                test_dataset, DataLoaderConfig(batch_size=1, shuffle=False, num_workers=0)
            )
            prediction_paths = run_three_d_inference(
                model,
                test_loader,
                source_paths,
                checkpoint_path,
                self.config.inference.three_d,
                self.config.output_dir,
            )
        else:
            slice_dicts, case_slice_counts = expand_to_slice_dicts(test_dicts)
            test_dataset = build_dataset(slice_dicts, build_transforms(self.config.transforms))
            test_loader = build_dataloader(
                test_dataset, DataLoaderConfig(batch_size=1, shuffle=False, num_workers=0)
            )
            prediction_paths = run_case_inference(
                model,
                test_loader,
                case_slice_counts,
                source_paths,
                checkpoint_path,
                self.config.inference.two_d,
                self.config.output_dir,
            )

        context.set("prediction_paths", prediction_paths)
        return context
