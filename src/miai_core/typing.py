"""Shared typing utilities used across MIAI packages.

Keeping these aliases in one place ensures every package accepts the same
shape of "path-like" and "config-like" arguments.
"""

from __future__ import annotations

import os
from typing import Any, Union

#: Anything that can be used as a filesystem path: a string or an
#: :class:`os.PathLike` (which includes :class:`pathlib.Path`).
StrPath = Union[str, "os.PathLike[str]"]

#: A plain, JSON/YAML-serializable mapping, as produced by
#: :mod:`miai_core.io` and consumed by :mod:`miai_core.config`.
JSONDict = dict[str, Any]
