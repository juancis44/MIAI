"""Configuration schema for building a MONAI transform pipeline from YAML."""

from __future__ import annotations

from typing import Any

from miai_core.config import MIAIBaseConfig


class TransformSpec(MIAIBaseConfig):
    """A single transform in a pipeline.

    Attributes:
        name: A short, registered name identifying the transform (see
            :data:`miai_transforms.compose.TRANSFORM_REGISTRY`), e.g.
            ``"load_image"`` or ``"rand_flip"``.
        params: Keyword arguments forwarded to the underlying MONAI
            transform's constructor.
    """

    name: str
    params: dict[str, Any] = {}


class TransformConfig(MIAIBaseConfig):
    """An ordered list of transforms, composed into a single pipeline.

    Example YAML::

        transforms:
          - name: load_image
            params: {keys: [image, label]}
          - name: rand_flip
            params: {keys: [image, label], prob: 0.5, spatial_axis: 0}
    """

    transforms: list[TransformSpec] = []
