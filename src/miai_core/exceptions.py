"""Exception hierarchy shared across the MIAI ecosystem.

All MIAI packages should raise subclasses of :class:`MIAIError` instead of
bare built-in exceptions, so calling code can catch MIAI-specific failures
without accidentally swallowing unrelated errors.
"""

from __future__ import annotations


class MIAIError(Exception):
    """Base class for all exceptions raised by MIAI packages."""


class ConfigError(MIAIError):
    """Raised when a configuration file or object is invalid or missing.

    Examples include a YAML file that fails schema validation, or a
    required configuration key that was not provided.
    """


class MIAIIOError(MIAIError):
    """Raised when reading or writing a file fails in a MIAI-specific way.

    This is distinct from Python's built-in :class:`OSError` /
    :class:`IOError`: it is raised for MIAI-level expectations (e.g. an
    unexpected file extension, a malformed serialized format) rather than
    low-level filesystem failures, which are allowed to propagate as-is.
    """


class ValidationError(MIAIError):
    """Raised when data does not satisfy an expected schema or invariant.

    Used for data-level checks (e.g. an image array with an unexpected
    number of dimensions) as opposed to configuration checks
    (:class:`ConfigError`).
    """


class NotFoundError(MIAIError):
    """Raised when a required resource (file, key, registered component) is
    missing.
    """
