"""Integration test for the concrete RegistrationStage."""

from pathlib import Path

import pytest

from conftest import make_offset_cube_volume
from miai_pipeline.context import PipelineContext
from miai_pipeline.exceptions import StageError
from miai_pipeline.stages.registration import RegistrationStage, RegistrationStageConfig
from miai_registration.register import RegistrationConfig

_FAST_REGISTRATION = RegistrationConfig(
    transform_type="rigid",
    metric="mean_squares",
    number_of_iterations=150,
    sampling_percentage=1.0,
    shrink_factors=(1,),
    smoothing_sigmas=(0.0,),
)


def test_registration_stage_aligns_cases_and_writes_outputs(tmp_path: Path) -> None:
    fixed_path = make_offset_cube_volume(tmp_path / "fixed", name="atlas")
    moving_path = make_offset_cube_volume(tmp_path / "moving", name="case0", offset=(3, 0, 0))
    label_path = make_offset_cube_volume(tmp_path / "moving", name="case0_label", offset=(3, 0, 0))

    ctx = PipelineContext()
    ctx.set("preprocessed_paths", [moving_path])
    ctx.set("label_paths", [label_path])

    stage = RegistrationStage(
        RegistrationStageConfig(
            fixed_image_path=str(fixed_path),
            output_dir=str(tmp_path / "registered"),
            transform_dir=str(tmp_path / "transforms"),
            registration=_FAST_REGISTRATION,
            label_context_key="label_paths",
        )
    )

    result = stage.run(ctx)

    registered_paths = result.require("registered_paths")
    transform_paths = result.require("transform_paths")
    registered_label_paths = result.require("registered_label_paths")

    assert len(registered_paths) == 1
    assert Path(registered_paths[0]).exists()
    assert Path(transform_paths[0]).exists()
    assert Path(registered_label_paths[0]).exists()


def test_registration_stage_empty_context_key_raises(tmp_path: Path) -> None:
    ctx = PipelineContext()
    ctx.set("preprocessed_paths", [])

    stage = RegistrationStage(
        RegistrationStageConfig(
            fixed_image_path="unused.nii.gz",
            output_dir=str(tmp_path / "registered"),
            transform_dir=str(tmp_path / "transforms"),
        )
    )

    with pytest.raises(StageError):
        stage.run(ctx)


def test_registration_stage_label_length_mismatch_raises(tmp_path: Path) -> None:
    fixed_path = make_offset_cube_volume(tmp_path / "fixed", name="atlas")
    moving_path = make_offset_cube_volume(tmp_path / "moving", name="case0", offset=(3, 0, 0))

    ctx = PipelineContext()
    ctx.set("preprocessed_paths", [moving_path])
    ctx.set("label_paths", [])

    stage = RegistrationStage(
        RegistrationStageConfig(
            fixed_image_path=str(fixed_path),
            output_dir=str(tmp_path / "registered"),
            transform_dir=str(tmp_path / "transforms"),
            registration=_FAST_REGISTRATION,
            label_context_key="label_paths",
        )
    )

    with pytest.raises(StageError):
        stage.run(ctx)
