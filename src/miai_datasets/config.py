"""Configuration for MONAI dataset/dataloader construction."""

from __future__ import annotations

from miai_core.config import MIAIBaseConfig


class DataLoaderConfig(MIAIBaseConfig):
    """Configuration for :func:`miai_datasets.loaders.build_dataloader`.

    Attributes:
        batch_size: Number of cases per batch.
        num_workers: Number of subprocess workers for data loading.
        shuffle: Whether to shuffle case order each epoch. Should be
            ``True`` for training and ``False`` for validation/inference.
        cache_rate: Fraction of cases to cache in memory via
            :class:`monai.data.CacheDataset` (``0.0`` disables caching).
    """

    batch_size: int = 1
    num_workers: int = 0
    shuffle: bool = False
    cache_rate: float = 0.0
