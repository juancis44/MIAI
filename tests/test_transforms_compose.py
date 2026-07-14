"""Tests for miai_transforms.compose."""

import pytest
from monai.transforms import Compose

from miai_transforms.compose import TRANSFORM_REGISTRY, build_transforms
from miai_transforms.config import TransformConfig, TransformSpec
from miai_transforms.exceptions import TransformError


def test_build_transforms_returns_compose_with_expected_length() -> None:
    config = TransformConfig(
        transforms=[
            TransformSpec(name="rand_flip", params={"keys": ["image"], "prob": 1.0}),
            TransformSpec(name="rand_rotate90", params={"keys": ["image"], "prob": 1.0}),
        ]
    )
    composed = build_transforms(config)
    assert isinstance(composed, Compose)
    assert len(composed.transforms) == 2


def test_build_transforms_empty_config_returns_empty_compose() -> None:
    composed = build_transforms(TransformConfig())
    assert isinstance(composed, Compose)
    assert len(composed.transforms) == 0


def test_build_transforms_unknown_name_raises_transform_error() -> None:
    config = TransformConfig(transforms=[TransformSpec(name="not_a_real_transform")])
    with pytest.raises(TransformError, match="Unknown transform"):
        build_transforms(config)


def test_build_transforms_invalid_params_raises_transform_error() -> None:
    config = TransformConfig(
        transforms=[TransformSpec(name="rand_flip", params={"not_a_real_kwarg": 1})]
    )
    with pytest.raises(TransformError, match="Invalid parameters"):
        build_transforms(config)


def test_registry_names_are_all_lowercase_snake_case() -> None:
    for name in TRANSFORM_REGISTRY:
        assert name == name.lower()
        assert " " not in name
