"""Tests for miai_core.utils."""

import random

from miai_core.utils import deep_update, set_seed, utc_timestamp


def test_set_seed_makes_random_reproducible() -> None:
    set_seed(123)
    first = [random.random() for _ in range(5)]
    set_seed(123)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_utc_timestamp_is_iso_format_string() -> None:
    ts = utc_timestamp()
    assert isinstance(ts, str)
    assert "T" in ts


def test_deep_update_merges_nested_dicts() -> None:
    base = {"model": {"name": "unet", "depth": 4}, "seed": 0}
    overrides = {"model": {"depth": 5}}

    result = deep_update(base, overrides)

    assert result == {"model": {"name": "unet", "depth": 5}, "seed": 0}


def test_deep_update_does_not_mutate_inputs() -> None:
    base = {"a": {"b": 1}}
    overrides = {"a": {"b": 2}}

    deep_update(base, overrides)

    assert base == {"a": {"b": 1}}
    assert overrides == {"a": {"b": 2}}


def test_deep_update_replaces_non_dict_values() -> None:
    base = {"tags": ["ct"]}
    overrides = {"tags": ["mri"]}

    result = deep_update(base, overrides)

    assert result == {"tags": ["mri"]}
