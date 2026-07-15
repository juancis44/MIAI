"""Tests for miai_registration.transform_io."""

from pathlib import Path

import SimpleITK as sitk

from miai_registration.transform_io import read_transform, write_transform


def test_write_and_read_transform_roundtrip(tmp_path: Path) -> None:
    transform = sitk.Euler3DTransform()
    transform.SetTranslation((1.0, 2.0, 3.0))

    out_path = write_transform(transform, tmp_path / "nested" / "t.tfm")

    assert out_path.exists()
    loaded = read_transform(out_path)
    assert loaded.GetParameters() == transform.GetParameters()
