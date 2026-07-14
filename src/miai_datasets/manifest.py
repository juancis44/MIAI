"""Turns a :mod:`miai_pipeline` manifest split into MONAI-style data dicts.

:class:`~miai_pipeline.stages.dataset.DatasetStage` writes a manifest
whose splits are lists of either plain path strings (image only) or
``{"image": ..., "label": ...}`` dicts (when the stage was configured
with a ``label_context_key``). This module normalizes either shape into
the ``list[dict[str, str]]`` MONAI's dataset/transform APIs expect.
"""

from __future__ import annotations

from typing import Any

from miai_datasets.exceptions import DatasetBuildError


def manifest_split_to_data_dicts(split: list[Any]) -> list[dict[str, str]]:
    """Normalize one manifest split into MONAI-style data dicts.

    Args:
        split: A manifest split (e.g. ``manifest["train"]``), as
            produced by :class:`~miai_pipeline.stages.dataset.DatasetStage`.
            Each entry is either a path string (image only) or a
            ``{"image": ..., "label": ...}`` mapping.

    Returns:
        One ``dict[str, str]`` per entry, always carrying an
        ``"image"`` key and, when present in the input, a ``"label"``
        key.

    Raises:
        DatasetBuildError: If an entry is a mapping without an
            ``"image"`` key, or is neither a string nor a mapping.
    """
    data_dicts: list[dict[str, str]] = []
    for entry in split:
        if isinstance(entry, str):
            data_dicts.append({"image": entry})
        elif isinstance(entry, dict):
            if "image" not in entry:
                raise DatasetBuildError(
                    f"Manifest entry is missing required key 'image': {entry!r}"
                )
            data_dicts.append({str(k): str(v) for k, v in entry.items()})
        else:
            raise DatasetBuildError(
                f"Manifest entry must be a path string or a mapping, got {type(entry).__name__}"
            )
    return data_dicts
