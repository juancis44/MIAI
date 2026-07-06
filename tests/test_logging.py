"""Tests for miai_core.logging."""

import logging

from miai_core.logging import configure_logging, get_logger


def test_get_logger_returns_namespaced_logger() -> None:
    logger = get_logger("mypackage.mymodule")
    assert logger.name == "miai.mypackage.mymodule"


def test_configure_logging_is_idempotent() -> None:
    configure_logging(level=logging.DEBUG)
    root = logging.getLogger("miai")
    handlers_after_first_call = list(root.handlers)

    configure_logging(level=logging.DEBUG)

    assert root.handlers == handlers_after_first_call


def test_configure_logging_force_resets_handlers() -> None:
    configure_logging(level=logging.INFO, force=True)
    root = logging.getLogger("miai")
    first_handler = root.handlers[0]

    configure_logging(level=logging.WARNING, force=True)

    assert root.level == logging.WARNING
    assert root.handlers[0] is not first_handler
