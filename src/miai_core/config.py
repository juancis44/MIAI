"""Configuration system: define experiments in YAML, not in Python code.

Every MIAI package that needs configuration should define a
:class:`pydantic.BaseModel` subclassing :class:`MIAIBaseConfig`, which adds
``from_yaml`` / ``to_yaml`` so the same object can be constructed from a
config file, validated, and serialized back out for reproducibility
(e.g. saved alongside experiment outputs).
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError as PydanticValidationError

from miai_core.exceptions import ConfigError
from miai_core.io import read_yaml, write_yaml
from miai_core.typing import StrPath

ConfigT = TypeVar("ConfigT", bound="MIAIBaseConfig")


class MIAIBaseConfig(BaseModel):
    """Base class for all MIAI configuration objects.

    Subclass this with typed fields describing an experiment, pipeline
    stage, or component, then load it from a YAML file with
    :meth:`from_yaml`::

        class TrainingConfig(MIAIBaseConfig):
            learning_rate: float
            batch_size: int

        config = TrainingConfig.from_yaml("configs/train.yaml")

    Unknown keys in the YAML file are rejected by default (``extra="forbid"``)
    so a typo in a config file fails loudly instead of being silently
    ignored.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def from_yaml(cls: type[ConfigT], path: StrPath) -> ConfigT:
        """Load and validate a configuration from a YAML file.

        Args:
            path: Path to a YAML file whose top-level keys match this
                config class's fields.

        Returns:
            A validated instance of ``cls``.

        Raises:
            NotFoundError: If ``path`` does not exist.
            MIAIIOError: If the file is not valid YAML.
            ConfigError: If the parsed contents do not satisfy this
                config's schema.
        """
        raw = read_yaml(path)
        try:
            return cls.model_validate(raw)
        except PydanticValidationError as exc:
            raise ConfigError(f"Invalid configuration in {path}:\n{exc}") from exc

    def to_yaml(self, path: StrPath) -> StrPath:
        """Serialize this configuration to a YAML file.

        Useful for saving the exact configuration used for a run alongside
        its outputs, so the run can be reproduced later.

        Args:
            path: Destination path.

        Returns:
            The destination path.
        """
        write_yaml(self.model_dump(mode="json"), path)
        return path
