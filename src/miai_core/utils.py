"""General-purpose utilities shared across MIAI packages.

Anything here should be small, dependency-light, and used by more than one
downstream package — otherwise it belongs in that package instead.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any

from miai_core.typing import JSONDict


def set_seed(seed: int) -> None:
    """Seed Python's and NumPy's random number generators for reproducibility.

    Does not seed PyTorch — packages that depend on ``miai-core`` and use
    PyTorch are responsible for also seeding it
    (``torch.manual_seed(seed)``), since ``miai-core`` itself does not
    depend on PyTorch.

    Args:
        seed: Seed value to use.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def utc_timestamp() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Useful for tagging experiment run directories and log entries with a
    consistent, sortable timestamp.

    Returns:
        Timestamp string, e.g. ``"2026-07-06T18:21:00+00:00"``.
    """
    return datetime.now(timezone.utc).isoformat()


def deep_update(base: JSONDict, overrides: JSONDict) -> JSONDict:
    """Recursively merge ``overrides`` into a copy of ``base``.

    Nested dictionaries are merged key by key rather than replaced
    wholesale, so overriding one field of a nested config section does not
    discard its siblings. Used to layer a CLI override or experiment
    variant on top of a base configuration file.

    Args:
        base: The base mapping.
        overrides: Values to overlay on top of ``base``.

    Returns:
        A new dictionary with ``overrides`` applied on top of ``base``.
        Neither input is mutated.
    """
    result: JSONDict = dict(base)
    for key, value in overrides.items():
        base_value: Any = result.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            result[key] = deep_update(base_value, value)
        else:
            result[key] = value
    return result
