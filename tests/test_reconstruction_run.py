"""Integration tests for run_kspace_reconstruction (SimpleITK I/O)."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import make_offset_cube_volume
from miai_reconstruction.exceptions import ReconstructionError
from miai_reconstruction.kspace import KSpaceReconstructionConfig, UndersamplingConfig
from miai_reconstruction.run import run_kspace_reconstruction


def test_run_kspace_reconstruction_writes_one_file_per_case(tmp_path: Path) -> None:
    image_path = make_offset_cube_volume(tmp_path / "data", name="case0", size=(8, 8, 8))

    paths = run_kspace_reconstruction(
        [str(image_path)], KSpaceReconstructionConfig(), None, str(tmp_path / "reconstructed")
    )

    assert len(paths) == 1
    assert paths[0].exists()


def test_run_kspace_reconstruction_with_undersampling_writes_file(tmp_path: Path) -> None:
    image_path = make_offset_cube_volume(tmp_path / "data", name="case0", size=(8, 8, 8))

    paths = run_kspace_reconstruction(
        [str(image_path)],
        KSpaceReconstructionConfig(),
        UndersamplingConfig(acceleration=4.0),
        str(tmp_path / "reconstructed"),
    )

    assert len(paths) == 1
    assert paths[0].exists()


def test_run_kspace_reconstruction_empty_paths_raises(tmp_path: Path) -> None:
    with pytest.raises(ReconstructionError):
        run_kspace_reconstruction([], KSpaceReconstructionConfig(), None, str(tmp_path / "out"))
