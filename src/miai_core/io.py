"""IO utilities: reading and writing YAML/JSON, and filesystem helpers.

These wrap the standard library / PyYAML with MIAI's exception hierarchy so
callers only need to catch :class:`miai_core.exceptions.MIAIIOError`
instead of a mix of ``OSError``, ``yaml.YAMLError``, and
``json.JSONDecodeError``.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from miai_core.exceptions import MIAIIOError, NotFoundError
from miai_core.typing import JSONDict, StrPath


def ensure_dir(path: StrPath) -> Path:
    """Create ``path`` (and parents) if it does not already exist.

    Args:
        path: Directory to create.

    Returns:
        The directory as a :class:`pathlib.Path`.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _require_exists(path: Path) -> None:
    if not path.exists():
        raise NotFoundError(f"File not found: {path}")


def read_yaml(path: StrPath) -> JSONDict:
    """Read a YAML file into a dictionary.

    Args:
        path: Path to a ``.yaml`` / ``.yml`` file.

    Returns:
        The parsed contents as a dictionary. An empty file returns ``{}``.

    Raises:
        NotFoundError: If ``path`` does not exist.
        MIAIIOError: If the file cannot be parsed as YAML, or does not
            deserialize to a mapping at the top level.
    """
    file_path = Path(path)
    _require_exists(file_path)
    try:
        content = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise MIAIIOError(f"Failed to parse YAML file {file_path}: {exc}") from exc

    if content is None:
        return {}
    if not isinstance(content, dict):
        raise MIAIIOError(
            f"Expected a mapping at the top level of {file_path}, got {type(content).__name__}"
        )
    return content


def write_yaml(data: JSONDict, path: StrPath) -> Path:
    """Write a dictionary to a YAML file, creating parent directories.

    Args:
        data: The data to serialize.
        path: Destination path.

    Returns:
        The destination path.

    Raises:
        MIAIIOError: If ``data`` cannot be serialized as YAML.
    """
    file_path = Path(path)
    ensure_dir(file_path.parent)
    try:
        serialized = yaml.safe_dump(data, sort_keys=False)
    except yaml.YAMLError as exc:
        raise MIAIIOError(f"Failed to serialize data to YAML for {file_path}: {exc}") from exc
    file_path.write_text(serialized, encoding="utf-8")
    return file_path


def read_json(path: StrPath) -> JSONDict:
    """Read a JSON file into a dictionary.

    Args:
        path: Path to a ``.json`` file.

    Returns:
        The parsed contents as a dictionary.

    Raises:
        NotFoundError: If ``path`` does not exist.
        MIAIIOError: If the file cannot be parsed as JSON, or does not
            deserialize to a mapping at the top level.
    """
    file_path = Path(path)
    _require_exists(file_path)
    try:
        content = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MIAIIOError(f"Failed to parse JSON file {file_path}: {exc}") from exc

    if not isinstance(content, dict):
        raise MIAIIOError(
            f"Expected a mapping at the top level of {file_path}, got {type(content).__name__}"
        )
    return content


def write_json(data: JSONDict, path: StrPath, *, indent: int = 2) -> Path:
    """Write a dictionary to a JSON file, creating parent directories.

    Args:
        data: The data to serialize.
        path: Destination path.
        indent: Indentation level passed to :func:`json.dumps`.

    Returns:
        The destination path.

    Raises:
        MIAIIOError: If ``data`` cannot be serialized as JSON.
    """
    file_path = Path(path)
    ensure_dir(file_path.parent)
    try:
        serialized = json.dumps(data, indent=indent)
    except TypeError as exc:
        raise MIAIIOError(f"Failed to serialize data to JSON for {file_path}: {exc}") from exc
    file_path.write_text(serialized, encoding="utf-8")
    return file_path
