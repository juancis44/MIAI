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
            paths to split — typically ``"preprocessed_paths"``, but
            can be set to ``"nifti_paths"`` to skip preprocessing.
    """

    manifest_path: str
    val_fraction: float = 0.2
    test_fraction: float = 0.0
    seed: int = 42
    context_key: str = "preprocessed_paths"


class DatasetStage(PipelineStage):
    """Split a list of case files into train/val/test and write a manifest.

    Reads:
        ``<config.context_key>`` (``list[Path]``, default
        ``"preprocessed_paths"``): the cases to split.

    Writes:
        ``manifest`` (``dict[str, list[str]]``): the split, keyed by
        ``"train"``, ``"val"``, ``"test"``.
        ``manifest_path`` (``str``): where the manifest was written.
    """

    name = "dataset"
    config_cls = DatasetConfig

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    def run(self, context: PipelineContext) -> PipelineContext:
        if self.config.val_fraction + self.config.test_fraction >= 1.0:
            raise ConfigError(
                "val_fraction + test_fraction must be less than 1.0 so at least "
                "one case remains for training."
            )

        cases: list[Path] = context.require(self.config.context_key)
        indices = list(range(len(cases)))
        random.Random(self.config.seed).shuffle(indices)

        n = len(indices)
        n_test = int(n * self.config.test_fraction)
        n_val = int(n * self.config.val_fraction)

        test_idx = indices[:n_test]
        val_idx = indices[n_test : n_test + n_val]
        train_idx = indices[n_test + n_val :]

        manifest = {
            "train": [str(cases[i]) for i in train_idx],
            "val": [str(cases[i]) for i in val_idx],
            "test": [str(cases[i]) for i in test_idx],
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
