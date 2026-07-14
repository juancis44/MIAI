"""Tests for miai_evaluation.evaluate."""

import json
from pathlib import Path

import pytest

from conftest import make_synthetic_volume_pair
from miai_evaluation.evaluate import evaluate_predictions
from miai_evaluation.exceptions import EvaluationError
from miai_evaluation.metrics import MetricsConfig


def test_evaluate_predictions_perfect_match_when_prediction_equals_ground_truth(
    tmp_path: Path,
) -> None:
    _, label_path = make_synthetic_volume_pair(tmp_path, name="case0")

    report = evaluate_predictions([str(label_path)], [str(label_path)], MetricsConfig())

    assert report["mean"]["dice"] == pytest.approx(1.0)
    assert report["mean"]["hausdorff_distance"] == pytest.approx(0.0)
    assert len(report["per_case"]) == 1
    assert report["per_case"][0]["case"] == label_path.name


def test_evaluate_predictions_writes_report_to_disk(tmp_path: Path) -> None:
    _, label_path = make_synthetic_volume_pair(tmp_path, name="case0")
    output_path = tmp_path / "report.json"

    evaluate_predictions([str(label_path)], [str(label_path)], MetricsConfig(), str(output_path))

    assert output_path.exists()
    saved = json.loads(output_path.read_text())
    assert saved["mean"]["dice"] == pytest.approx(1.0)


def test_evaluate_predictions_mismatched_lengths_raises(tmp_path: Path) -> None:
    _, label_path = make_synthetic_volume_pair(tmp_path, name="case0")

    with pytest.raises(EvaluationError):
        evaluate_predictions([str(label_path), str(label_path)], [str(label_path)], MetricsConfig())


def test_evaluate_predictions_empty_raises() -> None:
    with pytest.raises(EvaluationError):
        evaluate_predictions([], [], MetricsConfig())
