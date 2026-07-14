"""Exceptions specific to evaluation."""

from __future__ import annotations

from miai_core.exceptions import MIAIError


class EvaluationError(MIAIError):
    """Raised when predictions cannot be scored against ground truth.

    Examples include mismatched prediction/ground-truth counts, or an
    empty set of cases to evaluate.
    """
