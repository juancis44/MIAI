"""Tests for the miai-pipeline CLI (src/miai_pipeline/cli.py).

The argument-parsing helper and error paths are testable without torch
installed (miai_pipeline's top-level __init__.py doesn't import the
concrete stage implementations); the "run"/"validate"/"list-stages"
subcommands themselves do need torch, since building a Pipeline lazily
imports miai_pipeline.stages, which eagerly imports every concrete
stage -- including the torch-dependent ones (diffusion, foundation
models, export, reconstruction, visualization).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import make_dicom_series
from miai_pipeline.cli import _parse_set_value, main


def test_parse_set_value_splits_on_first_equals() -> None:
    assert _parse_set_value("dicom_dir=data/raw") == ("dicom_dir", "data/raw")
    assert _parse_set_value("key=a=b") == ("key", "a=b")


def test_parse_set_value_missing_equals_raises() -> None:
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        _parse_set_value("no_equals_sign")


def test_parse_set_value_empty_key_raises() -> None:
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        _parse_set_value("=value")


def test_list_stages_prints_registered_names(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["list-stages"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "dicom_to_nifti" in captured.out
    assert "training" in captured.out


def test_validate_valid_config_prints_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        "stages:\n  - type: dicom_to_nifti\n    params:\n      output_dir: "
        f"{tmp_path / 'nifti'}\n"
    )

    exit_code = main(["validate", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "1 stage(s)" in captured.out
    assert "dicom_to_nifti" in captured.out


def test_validate_unknown_stage_type_returns_one(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("stages:\n  - type: not_a_real_stage\n    params: {}\n")

    exit_code = main(["validate", str(config_path)])

    assert exit_code == 1


def test_run_executes_pipeline_and_writes_output(tmp_path: Path) -> None:
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()
    make_dicom_series(dicom_dir, num_slices=4, rows=16, columns=16)

    output_dir = tmp_path / "nifti"
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        f"stages:\n  - type: dicom_to_nifti\n    params:\n      output_dir: {output_dir}\n"
    )

    exit_code = main(["run", str(config_path), "--set", f"dicom_dir={dicom_dir}"])

    assert exit_code == 0
    assert any(output_dir.iterdir())
