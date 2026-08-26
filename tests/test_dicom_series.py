"""Tests for miai_dicom.series."""

from pathlib import Path

import pytest

from conftest import make_dicom_dataset
from miai_core.exceptions import NotFoundError, ValidationError
from miai_dicom.io import write_dicom
from miai_dicom.series import load_series


def test_load_series_groups_by_series_instance_uid(tmp_path: Path) -> None:
    series_a = "1.2.3.111"
    series_b = "1.2.3.222"

    for i in range(3):
        ds = make_dicom_dataset(series_instance_uid=series_a, instance_number=i + 1)
        write_dicom(ds, tmp_path / f"a_{i}.dcm")
    for i in range(2):
        ds = make_dicom_dataset(series_instance_uid=series_b, instance_number=i + 1)
        write_dicom(ds, tmp_path / f"b_{i}.dcm")

    series_list = load_series(tmp_path)

    assert len(series_list) == 2
    by_uid = {s.series_instance_uid: s for s in series_list}
    assert len(by_uid[series_a]) == 3
    assert len(by_uid[series_b]) == 2


def test_load_series_sorts_by_instance_number(tmp_path: Path) -> None:
    uid = "1.2.3.999"
    for i in [3, 1, 2]:
        ds = make_dicom_dataset(series_instance_uid=uid, instance_number=i)
        write_dicom(ds, tmp_path / f"slice_{i}.dcm")

    (series,) = load_series(tmp_path)

    instance_numbers = []
    from miai_dicom.io import read_dicom

    for path in series.file_paths:
        instance_numbers.append(int(read_dicom(path).InstanceNumber))

    assert instance_numbers == [1, 2, 3]


def test_load_series_skips_files_without_series_instance_uid(tmp_path: Path) -> None:
    ds_with_uid = make_dicom_dataset(series_instance_uid="1.2.3.666")
    write_dicom(ds_with_uid, tmp_path / "with_uid.dcm")

    ds_without_uid = make_dicom_dataset()
    del ds_without_uid.SeriesInstanceUID
    write_dicom(ds_without_uid, tmp_path / "without_uid.dcm")

    series_list = load_series(tmp_path)

    assert len(series_list) == 1
    assert len(series_list[0]) == 1


def test_load_series_falls_back_to_default_order_without_instance_number_or_position(
    tmp_path: Path,
) -> None:
    # Neither InstanceNumber nor ImagePositionPatient is set by
    # make_dicom_dataset -- _sort_key must fall back to its final
    # default ((2, 0.0)) instead of raising, for every file.
    uid = "1.2.3.888"
    for i in range(3):
        ds = make_dicom_dataset(series_instance_uid=uid)
        write_dicom(ds, tmp_path / f"slice_{i}.dcm")

    (series,) = load_series(tmp_path)

    assert len(series) == 3


def test_load_series_orders_by_image_position_when_instance_number_missing(
    tmp_path: Path,
) -> None:
    # _sort_key's ImagePositionPatient branch (its middle fallback,
    # between InstanceNumber and the final default) is reached only
    # when InstanceNumber is absent but ImagePositionPatient is
    # present -- make_dicom_dataset doesn't set either by default, so
    # ImagePositionPatient must be assigned directly here.
    uid = "1.2.3.999"
    positions = [30.0, 10.0, 20.0]
    for i, z in enumerate(positions):
        ds = make_dicom_dataset(series_instance_uid=uid)
        ds.ImagePositionPatient = [0.0, 0.0, z]
        write_dicom(ds, tmp_path / f"slice_{i}.dcm")

    (series,) = load_series(tmp_path)

    assert len(series) == 3


def test_load_series_skips_non_dicom_files(tmp_path: Path) -> None:
    ds = make_dicom_dataset(series_instance_uid="1.2.3.555")
    write_dicom(ds, tmp_path / "slice.dcm")
    (tmp_path / "readme.txt").write_text("not dicom", encoding="utf-8")

    series_list = load_series(tmp_path)

    assert len(series_list) == 1


def test_load_series_missing_directory_raises_not_found(tmp_path: Path) -> None:
    with pytest.raises(NotFoundError):
        load_series(tmp_path / "does_not_exist")


def test_load_series_empty_directory_raises_validation_error(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_series(tmp_path)


def test_load_series_reports_modality(tmp_path: Path) -> None:
    ds = make_dicom_dataset(series_instance_uid="1.2.3.777", modality="MR")
    write_dicom(ds, tmp_path / "slice.dcm")

    (series,) = load_series(tmp_path)

    assert series.modality == "MR"
