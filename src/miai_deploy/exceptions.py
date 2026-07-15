"""Exceptions raised by :mod:`miai_deploy`."""

from __future__ import annotations

from miai_core.exceptions import MIAIError


class DeployError(MIAIError):
    """Raised for model-export/bundling configuration or usage errors.

    Examples include an unknown export format reaching runtime code
    (bypassing pydantic's own validation, e.g. via
    ``model_copy(update=...)``), or a checkpoint that does not match
    the given model architecture.
    """
