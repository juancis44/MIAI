"""The mutable context object threaded through a pipeline run.

Each stage reads inputs it needs from the context and writes its outputs
back into it, so stages stay decoupled: a stage only needs to agree with
its neighbors on context *key names*, not on being called in a specific
class hierarchy or module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from miai_core.exceptions import NotFoundError


@dataclass
class PipelineContext:
    """Key-value store passed between pipeline stages.

    Unlike :class:`miai_core.typing.JSONDict`, values are not required to
    be JSON-serializable: a context commonly carries in-memory objects
    such as file paths, arrays, or (in later phases) model checkpoints.
    """

    _data: dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key``, overwriting any existing value."""
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for ``key``, or ``default`` if not present."""
        return self._data.get(key, default)

    def require(self, key: str) -> Any:
        """Return the value for ``key``.

        Args:
            key: The context key a stage depends on.

        Returns:
            The stored value.

        Raises:
            NotFoundError: If ``key`` has not been set by an earlier
                stage. The message names the missing key so a
                misconfigured pipeline (e.g. stages in the wrong order)
                fails with an actionable error.
        """
        if key not in self._data:
            raise NotFoundError(
                f"Pipeline context is missing required key '{key}'. "
                "Check that an earlier stage sets it before this stage runs."
            )
        return self._data[key]

    def __contains__(self, key: object) -> bool:
        return key in self._data
