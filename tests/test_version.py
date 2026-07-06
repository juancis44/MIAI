"""Smoke test to confirm the package imports and exposes a version."""

from miai_core import __version__


def test_version_is_string() -> None:
    """The package version should be a non-empty string."""
    assert isinstance(__version__, str)
    assert __version__
