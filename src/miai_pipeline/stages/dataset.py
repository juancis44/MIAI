"""Manifest generation and train/val/test split stage."""

from __future__ import annotations

import random
from pathlib import Path

from miai_core.config import MIAIBaseConfig
from miai_core.exceptions import ConfigError
from miai_core.io import write_json
from miai_core.logging import get_logger
from miai_pipeline.context import PipelineContext
from miai_pipeline.stage import PipelineStage

logger = get_logger(__name__)


class DatasetConfig(MIAIBaseConfig):
    """Configuration for :class:`DatasetStage`.

    Attributes:
        manifest_path: Where to write the JSON manifest of the split.
        val_fraction: Fraction of cases assigned to the validation
            split.
        test_fraction: Fraction of cases assigned to the test split.
            The remainder goes to the training split.
        seed: Random seed for the shuffle, so the split is reproducible.
        context_key: Which context key holds the list of case file
            paths to split -- typically ``"preprocessed_paths"``, but
            can be set to ``"nifti_paths"`` to skip preprocessing.
        label_context_key: Optional context key holding a list of label
            (ground truth) file paths, aligned index-for-index with
            ``context_key``. If set, each manifest entry becomes a
            ``{"image": ..., "label": ...}`` mapping instead of a plain
            path string, which :mod:`miai_datasets` reads for
            supervised training/evaluation.
    """

    manifest_path: str
    val_fraction: float = 0.2
    test_fraction: float = 0.0
    seed: int = 42
    context_key: str = "preprocessed_paths"
    label_context_key: str | None = None


class DatasetStage(PipelineStage):
    """Split a list of case files into train/val/test and write a manifest.

    Reads:
        ``<config.context_key>`` (``list[Path]``, default
        ``"preprocessed_paths"``): the cases to split.
        ``<config.label_context_key>`` (``list[Path]``, optional): the
        label file for each case, same order as ``<config.context_key>``.

    Writes:
        ``manifest`` (``dict[str, list]``): the split, keyed by
        ``"train"``, ``"val"``, ``"test"``. Each entry is a path string,
        or -- when ``config.label_context_key`` is set -- a
        ``{"image": ..., "label": ...}`` mapping.
        ``manifest_path`` (``str``): where the manifest was written.
    """

    name = "dataset"
    config_cls = DatasetConfig

    def __init__(self, config: DatasetConfig) -> None:
        """Store this stage's configuration."""
        self.config = config

    def run(self, context: PipelineContext) -> PipelineContext:
        """Run the stage; see the class docstring for its read/write contract."""
        if self.config.val_fraction + self.config.test_fraction >= 1.0:
            raise ConfigError(
                "val_fraction + test_fraction must be less than 1.0 so at least "
                "one case remains for training."
            )

        cases: list[Path] = context.require(self.config.context_key)

        labels: list[Path] | None = None
        if self.config.label_context_key is not None:
            labels = context.require(self.config.label_context_key)
            if len(labels) != len(cases):
                raise ConfigError(
                    f"'{self.config.label_context_key}' has {len(labels)} entries but "
                    f"'{self.config.context_key}' has {len(cases)}; they must be aligned "
                    "one label per case."
                )

        indices = list(range(len(cases)))
        random.Random(self.config.seed).shuffle(indices)

        n = len(indices)
        n_test = int(n * self.config.test_fraction)
        n_val = int(n * self.config.val_fraction)

        test_idx = indices[:n_test]
        val_idx = indices[n_test : n_test + n_val]
        train_idx = indices[n_test + n_val :]

        def _entries(idxs: list[int]) -> list[object]:
            if labels is None:
                return [str(cases[i]) for i in idxs]
            return [{"image": str(cases[i]), "label": str(labels[i])} for i in idxs]

        manifest = {
            "train": _entries(train_idx),
            "val": _entries(val_idx),
            "test": _entries(test_idx),
        }
        logger.info(
            "Dataset split: %d train, %d val, %d test",
            len(manifest["train"]),
            len(manifest["val"]),
            len(manifest["test"]),
        )

        write_json(manifest, self.config.manifest_path)
        context.set("manifest", manifest)
        context.set("manifest_path", self.config.manifest_path)
        return context
