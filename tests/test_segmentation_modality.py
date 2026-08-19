"""Tests for miai_segmentation.modality (cross-modality dispatch)."""

import pytest

from miai_segmentation.exceptions import SegmentationError
from miai_segmentation.modality import (
    SegmentationInferenceConfig,
    SegmentationModalityConfig,
    build_model_for_modality,
    inference_config_for_modality,
)
from miai_segmentation.three_d.infer import InferenceConfig as ThreeDInferenceConfig
from miai_segmentation.two_d.infer import InferenceConfig as TwoDInferenceConfig
from miai_segmentation.two_d.models import ArchitectureConfig as TwoDArchitectureConfig
from miai_segmentation.two_half_d.models import ArchitectureConfig as TwoHalfDArchitectureConfig


def test_build_model_for_modality_three_d_default() -> None:
    model = build_model_for_modality(SegmentationModalityConfig())
    assert model.__class__.__name__ in {"UNet", "SegResNet", "AttentionUnet"}


def test_build_model_for_modality_two_d() -> None:
    config = SegmentationModalityConfig(modality="two_d", two_d=TwoDArchitectureConfig())
    model = build_model_for_modality(config)
    assert model is not None


def test_build_model_for_modality_two_half_d() -> None:
    config = SegmentationModalityConfig(
        modality="two_half_d", two_half_d=TwoHalfDArchitectureConfig()
    )
    model = build_model_for_modality(config)
    assert model is not None


def test_build_model_for_modality_unknown_modality_raises() -> None:
    config = SegmentationModalityConfig.model_construct(modality="bogus")
    with pytest.raises(SegmentationError, match="Unknown segmentation modality"):
        build_model_for_modality(config)


def test_inference_config_for_modality_three_d() -> None:
    config = SegmentationInferenceConfig(three_d=ThreeDInferenceConfig(roi_size=(4, 4, 4)))
    selected = inference_config_for_modality("three_d", config)
    assert selected is config.three_d


def test_inference_config_for_modality_two_d() -> None:
    config = SegmentationInferenceConfig(two_d=TwoDInferenceConfig(roi_size=(4, 4)))
    selected = inference_config_for_modality("two_d", config)
    assert selected is config.two_d


def test_inference_config_for_modality_two_half_d_uses_two_d_config() -> None:
    config = SegmentationInferenceConfig(two_d=TwoDInferenceConfig(roi_size=(4, 4)))
    selected = inference_config_for_modality("two_half_d", config)
    assert selected is config.two_d


def test_inference_config_for_modality_unknown_modality_raises() -> None:
    with pytest.raises(SegmentationError, match="Unknown segmentation modality"):
        inference_config_for_modality("bogus", SegmentationInferenceConfig())  # type: ignore[arg-type]
