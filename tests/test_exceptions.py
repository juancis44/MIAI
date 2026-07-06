"""Tests for miai_core.exceptions hierarchy."""

import pytest

from miai_core.exceptions import (
    ConfigError,
    MIAIError,
    MIAIIOError,
    NotFoundError,
    ValidationError,
)


@pytest.mark.parametrize("exc_cls", [ConfigError, MIAIIOError, ValidationError, NotFoundError])
def test_all_core_exceptions_subclass_miai_error(exc_cls: type[Exception]) -> None:
    assert issubclass(exc_cls, MIAIError)


def test_miai_error_subclasses_exception() -> None:
    assert issubclass(MIAIError, Exception)


def test_exceptions_carry_message() -> None:
    with pytest.raises(ConfigError, match="bad config"):
        raise ConfigError("bad config")
