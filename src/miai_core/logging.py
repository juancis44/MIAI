"""Consistent logging setup for MIAI packages.

Every MIAI package should call :func:`get_logger` instead of calling
``logging.getLogger`` directly, so log formatting stays consistent across
the ecosystem regardless of which package emits the message.
"""

from __future__ import annotations

import logging
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False


def configure_logging(level: int | str = logging.INFO, *, force: bool = False) -> None:
    """Configure the root ``miai`` logger with a consistent format.

    Safe to call multiple times: only the first call (or a later call with
    ``force=True``) has an effect, so importing several MIAI packages that
    each call this at import time will not duplicate handlers.

    Args:
        level: Logging level, e.g. ``logging.INFO`` or ``"DEBUG"``.
        force: Reconfigure even if logging was already configured.

    Raises:
        ValueError: If ``level`` is a string that is not a valid logging
            level name.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    root = logging.getLogger("miai")
    root.handlers.clear()

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under ``miai``.

    Args:
        name: Usually ``__name__`` of the calling module. The returned
            logger is created as a child of the ``miai`` logger
            (``miai.<name>``) so a single call to
            :func:`configure_logging` controls formatting for every MIAI
            package.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    configure_logging()
    return logging.getLogger(f"miai.{name}")
