"""Tests for miai_datasets.loaders."""

import pytest
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import Compose

from miai_datasets.config import DataLoaderConfig
from miai_datasets.exceptions import DatasetBuildError
from miai_datasets.loaders import build_dataloader, build_dataset


def test_build_dataset_plain_dataset_by_default() -> None:
    dataset = build_dataset([{"image": "a"}, {"image": "b"}], Compose([]))
    assert isinstance(dataset, Dataset)
    assert not isinstance(dataset, CacheDataset)
    assert len(dataset) == 2
    assert dataset[0] == {"image": "a"}


def test_build_dataset_cache_dataset_when_cache_rate_positive() -> None:
    dataset = build_dataset([{"image": "a"}], Compose([]), cache_rate=1.0)
    assert isinstance(dataset, CacheDataset)


def test_build_dataset_empty_raises() -> None:
    with pytest.raises(DatasetBuildError, match="empty"):
        build_dataset([], Compose([]))


def test_build_dataloader_respects_config() -> None:
    dataset = build_dataset([{"image": "a"}, {"image": "b"}], Compose([]))
    loader = build_dataloader(dataset, DataLoaderConfig(batch_size=2, shuffle=False))
    assert isinstance(loader, DataLoader)
    batches = list(loader)
    assert len(batches) == 1
    assert list(batches[0]["image"]) == ["a", "b"]
