"""MIAI Datasets: turns a miai-pipeline manifest into MONAI datasets/loaders.

Bridges :class:`~miai_pipeline.stages.dataset.DatasetStage`'s JSON
manifest and :mod:`miai_transforms` transform pipelines into
:class:`monai.data.Dataset` / :class:`monai.data.DataLoader` objects
ready for training or inference.
"""

from miai_datasets.config import DataLoaderConfig
from miai_datasets.exceptions import DatasetBuildError
from miai_datasets.loaders import build_dataloader, build_dataset
from miai_datasets.manifest import manifest_split_to_data_dicts

__version__ = "0.1.0"

__all__ = [
    "build_dataset",
    "build_dataloader",
    "DataLoaderConfig",
    "DatasetBuildError",
    "manifest_split_to_data_dicts",
    "__version__",
]
