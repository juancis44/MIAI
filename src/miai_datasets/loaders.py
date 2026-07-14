"""Builds MONAI datasets and dataloaders from normalized data dicts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from monai.data import CacheDataset, DataLoader, Dataset

from miai_datasets.config import DataLoaderConfig
from miai_datasets.exceptions import DatasetBuildError


def build_dataset(
    data_dicts: list[dict[str, str]],
    transforms: Callable[[Any], Any],
    *,
    cache_rate: float = 0.0,
) -> Dataset:
    """Build a MONAI dataset over a list of data dicts.

    Args:
        data_dicts: Normalized manifest entries, as produced by
            :func:`miai_datasets.manifest.manifest_split_to_data_dicts`.
        transforms: A composed transform pipeline (e.g. from
            :func:`miai_transforms.build_transforms`) applied to each
            item.
        cache_rate: Fraction of items to cache in memory after the
            first epoch. ``0.0`` (default) uses a plain
            :class:`monai.data.Dataset`; any value above ``0.0`` uses a
            :class:`monai.data.CacheDataset`, which trades startup time
            and memory for faster subsequent epochs.

    Returns:
        A MONAI dataset, indexable and iterable like any
        :class:`torch.utils.data.Dataset`.

    Raises:
        DatasetBuildError: If ``data_dicts`` is empty.
    """
    if not data_dicts:
        raise DatasetBuildError("Cannot build a dataset from an empty manifest split.")
    if cache_rate > 0.0:
        return CacheDataset(data=data_dicts, transform=transforms, cache_rate=cache_rate)
    return Dataset(data=data_dicts, transform=transforms)


def build_dataloader(dataset: Dataset, config: DataLoaderConfig) -> DataLoader:
    """Wrap a MONAI dataset in a :class:`monai.data.DataLoader`.

    Args:
        dataset: A dataset built by :func:`build_dataset`.
        config: Batching/loading configuration.

    Returns:
        A :class:`monai.data.DataLoader` (a thin, list-collating
        subclass of :class:`torch.utils.data.DataLoader`).
    """
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
    )
