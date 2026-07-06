"""MIAI Core: common utilities shared across the MIAI ecosystem.

Provides the configuration system, logging setup, IO helpers, the shared
exception hierarchy, typing aliases, and small general-purpose utilities
that every other MIAI package builds on. See docs/roadmap.md in the
repository root for what's implemented in later phases.
"""

from miai_core.config import MIAIBaseConfig
from miai_core.exceptions import (
    ConfigError,
    MIAIError,
    MIAIIOError,
    NotFoundError,
    ValidationError,
)
from miai_core.logging import configure_logging, get_logger

__version__ = "0.2.0"

__all__ = [
    "MIAIBaseConfig",
    "MIAIError",
    "ConfigError",
    "MIAIIOError",
    "NotFoundError",
    "ValidationError",
    "configure_logging",
    "get_logger",
    "__version__",
]
