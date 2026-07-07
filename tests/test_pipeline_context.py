"""Tests for miai_pipeline.context."""

import pytest

from miai_core.exceptions import NotFoundError
from miai_pipeline.context import PipelineContext


def test_set_and_get() -> None:
    ctx = PipelineContext()
    ctx.set("key", "value")
    assert ctx.get("key") == "value"


def test_get_returns_default_when_missing() -> None:
    ctx = PipelineContext()
    assert ctx.get("missing", "fallback") == "fallback"
    assert ctx.get("missing") is None


def test_require_returns_value_when_present() -> None:
    ctx = PipelineContext()
    ctx.set("key", 42)
    assert ctx.require("key") == 42


def test_require_raises_not_found_when_missing() -> None:
    ctx = PipelineContext()
    with pytest.raises(NotFoundError, match="key"):
        ctx.require("key")


def test_contains() -> None:
    ctx = PipelineContext()
    assert "key" not in ctx
    ctx.set("key", 1)
    assert "key" in ctx
