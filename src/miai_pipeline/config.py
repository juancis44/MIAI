"""Configuration schema for building a Pipeline from a YAML file.

A pipeline is defined as an ordered list of stages, each identified by a
``type`` string that must be registered in
:data:`miai_pipeline.stages.STAGE_REGISTRY`, with stage-specific
``params`` validated against that stage's own config class. This keeps
the top-level pipeline config generic while still giving each stage
strict, typed validation of its own parameters.
"""

from __future__ import annotations

from typing import Any

from miai_core.config import MIAIBaseConfig


class StageConfig(MIAIBaseConfig):
    """Configuration for a single stage within a pipeline config file."""

    type: str
    params: dict[str, Any] = {}


class PipelineConfig(MIAIBaseConfig):
    """Top-level configuration for an entire pipeline.

    Example YAML::

        stages:
          - type: dicom_to_nifti
            params:
              output_dir: data/nifti
          - type: preprocessing
            params:
              target_spacing: [1.0, 1.0, 1.0]
              normalization: zscore
    """

    stages: list[StageConfig]
