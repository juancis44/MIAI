"""Tests for miai_core.config."""

from pathlib import Path

import pydantic
import pytest

from miai_core.config import MIAIBaseConfig
from miai_core.exceptions import ConfigError, NotFoundError


class TrainingConfig(MIAIBaseConfig):
    learning_rate: float
    batch_size: int
    seed: int = 0


def test_from_yaml_loads_valid_config(tmp_path: Path) -> None:
    path = tmp_path / "train.yaml"
    path.write_text("learning_rate: 0.001\nbatch_size: 32\n", encoding="utf-8")

    config = TrainingConfig.from_yaml(path)

    assert config.learning_rate == 0.001
    assert config.batch_size == 32
    assert config.seed == 0


def test_from_yaml_missing_file_raises_not_found(tmp_path: Path) -> None:
    with pytest.raises(NotFoundError):
        TrainingConfig.from_yaml(tmp_path / "missing.yaml")


def test_from_yaml_missing_required_field_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "train.yaml"
    path.write_text("learning_rate: 0.001\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        TrainingConfig.from_yaml(path)


def test_from_yaml_unknown_field_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "train.yaml"
    path.write_text("learning_rate: 0.001\nbatch_size: 32\ntypo_field: true\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        TrainingConfig.from_yaml(path)


def test_config_is_frozen() -> None:
    config = TrainingConfig(learning_rate=0.01, batch_size=8)
    with pytest.raises(pydantic.ValidationError):
        config.batch_size = 16


def test_to_yaml_roundtrip(tmp_path: Path) -> None:
    config = TrainingConfig(learning_rate=0.01, batch_size=8, seed=7)
    path = tmp_path / "out.yaml"

    config.to_yaml(path)

    assert TrainingConfig.from_yaml(path) == config
