"""Tests for miai_core.io."""

from pathlib import Path

import pytest

from miai_core.exceptions import MIAIIOError, NotFoundError
from miai_core.io import ensure_dir, read_json, read_yaml, write_json, write_yaml


def test_ensure_dir_creates_nested_directory(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert result == target
    assert target.is_dir()


def test_write_then_read_yaml_roundtrip(tmp_path: Path) -> None:
    data = {"learning_rate": 0.001, "layers": [16, 32, 64]}
    path = tmp_path / "config.yaml"
    write_yaml(data, path)
    assert read_yaml(path) == data


def test_write_then_read_json_roundtrip(tmp_path: Path) -> None:
    data = {"seed": 42, "modalities": ["CT", "MRI"]}
    path = tmp_path / "config.json"
    write_json(data, path)
    assert read_json(path) == data


def test_read_yaml_missing_file_raises_not_found(tmp_path: Path) -> None:
    with pytest.raises(NotFoundError):
        read_yaml(tmp_path / "missing.yaml")


def test_read_yaml_empty_file_returns_empty_dict(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert read_yaml(path) == {}


def test_read_yaml_non_mapping_raises_io_error(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(MIAIIOError):
        read_yaml(path)


def test_read_yaml_malformed_syntax_raises_io_error(tmp_path: Path) -> None:
    path = tmp_path / "malformed.yaml"
    # An unclosed flow sequence is a genuine YAML syntax error (a
    # yaml.YAMLError), distinct from the "valid YAML, wrong top-level
    # type" case covered by test_read_yaml_non_mapping_raises_io_error.
    path.write_text("key: [1, 2\n", encoding="utf-8")
    with pytest.raises(MIAIIOError):
        read_yaml(path)


def test_write_yaml_unserializable_data_raises_io_error(tmp_path: Path) -> None:
    # PyYAML's safe_dump has no representer for an arbitrary object, so
    # this raises yaml.representer.RepresenterError (a YAMLError).
    with pytest.raises(MIAIIOError):
        write_yaml({"bad": object()}, tmp_path / "config.yaml")


def test_read_json_non_mapping_raises_io_error(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(MIAIIOError):
        read_json(path)


def test_write_json_unserializable_data_raises_io_error(tmp_path: Path) -> None:
    # json.dumps has no default encoder for an arbitrary object, so
    # this raises TypeError.
    with pytest.raises(MIAIIOError):
        write_json({"bad": object()}, tmp_path / "config.json")


def test_read_json_missing_file_raises_not_found(tmp_path: Path) -> None:
    with pytest.raises(NotFoundError):
        read_json(tmp_path / "missing.json")


def test_read_json_invalid_json_raises_io_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(MIAIIOError):
        read_json(path)


def test_write_yaml_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "config.yaml"
    write_yaml({"a": 1}, path)
    assert path.exists()
