"""Visualize the ACDC real-data validation effort's results.

``examples/validate_acdc.py`` has run twelve iterations (see
``docs/real_data_validation.md``) and every one of them was reported
as plain text and markdown tables -- no plot, curve, or image was ever
produced for any of it. This script closes that gap using
``miai_visualization`` (a real, tested package that until now was only
wired into the generic ``examples/segmentation_pipeline.py`` demo, via
``VisualizationStage``, never into the ACDC effort) against outputs
the completed iterations already wrote to disk -- no retraining, no
new pipeline run.

Four kinds of plot, one function each:

1. ``plot_all_training_curves`` -- parses the text training logs
   iterations 8/10/11/12 wrote (``miai_segmentation.three_d.train``
   doesn't emit a CSV log itself, only INFO-level text lines) into a
   per-iteration ``epoch,train_loss,val_dice`` CSV, then calls
   :func:`miai_visualization.curves.plot_training_curves`: one chart
   dedicated to the twelfth iteration alone (showing its epoch-23
   collapse in context), one comparing best-validation-Dice-yet
   iteration (12th) against the reference baseline (8th) and the two
   attention-gate iterations (10th/11th) on the same axes.
2. ``plot_all_case_comparisons`` -- for a fixed handful of test cases
   (the two weakest twelfth-iteration cases, `patient142_frame12` and
   `patient086_frame08`, plus a representative case near the median,
   `patient001_frame01`) -- uses
   :func:`miai_visualization.comparison.plot_comparison` to show the
   ground-truth and predicted label maps side by side with an absolute
   difference map, and :func:`miai_visualization.slices.plot_slice` to
   overlay the prediction on the source MRI for anatomical context.
   Run once per iteration (twelfth and, for comparison, the eighth/
   baseline) against the *same* cases and frames -- the eighth and
   twelfth iterations share an identical 60-case test set (same
   150-patient list, same `seed=42` split), so the two sets of plots
   are directly comparable frame for frame.
3. ``plot_macro_dice_bar_chart`` / ``plot_per_class_dice_box`` -- read
   an iteration's ``evaluation_report.json`` (still on disk under
   ``/tmp/deliverables/``, per the project's established naming).
   ``plot_macro_dice_bar_chart`` plots macro test Dice across
   iterations 8-12 as a bar chart; ``plot_per_class_dice_box`` plots
   one iteration's per-case Dice split by class (RV/Myo/LV) as a box
   plot, via :func:`miai_visualization.metrics.plot_metric_summary`
   (``kind="bar"``/``"box"``) -- run for both the twelfth iteration and
   the eighth/baseline, so the two class-wise distributions can be
   compared directly.
4. ``run_qc_montages`` -- runs the actual
   :class:`~miai_pipeline.stages.visualization.VisualizationStage`
   (unmodified, the same class ``examples/segmentation_pipeline.py``
   already uses) over every one of an iteration's 60 held-out
   test-set images, writing one QC slice-montage PNG per case. This is
   the same code path ``examples/validate_acdc.py --visualize`` now
   also runs at the end of a live pipeline run (see that script's
   module docstring) -- this function demonstrates it end to end
   against a completed run without waiting for a new one. Run for both
   the twelfth iteration and the eighth/baseline.
5. ``plot_points_of_interest_grid`` -- a single 3x3 figure per
   iteration meant to surface interesting cases at a glance rather
   than requiring one PNG per case: each row is one representative
   test case by macro Dice (best, closest-to-median, worst), each
   column is one of the three segmentation views ground truth /
   prediction / |difference| (the same three panels
   ``plot_all_case_comparisons`` already produces per case, here
   assembled into one grid instead of nine separate files). Run for
   both the twelfth iteration and the eighth/baseline.

This script hardcodes the specific `/tmp/...` paths this sandbox
session's twelve ACDC iterations wrote their logs/outputs/reports to
(see ``docs/real_data_validation.md`` for the full list) -- it is a
one-off analysis script for this validation effort, not a general
reusable example, so unlike ``examples/segmentation_pipeline.py`` it
does not generate its own synthetic data.

Run:
    python examples/visualize_acdc_results.py --output-dir /tmp/acdc_visualizations
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from miai_pipeline.context import PipelineContext
from miai_pipeline.stages.visualization import VisualizationStage, VisualizationStageConfig
from miai_visualization.comparison import PlotComparisonConfig, plot_comparison
from miai_visualization.curves import PlotTrainingCurvesConfig, plot_training_curves
from miai_visualization.metrics import PlotMetricSummaryConfig, plot_metric_summary
from miai_visualization.slices import PlotSliceConfig, plot_slice

#: Where each iteration's text training log lives in this sandbox
#: session (see docs/real_data_validation.md's "Long background runs"
#: notes in project memory for the full list; only the four iterations
#: this comparison needs are listed here).
_TRAINING_LOGS = {
    "8th (baseline)": Path("/tmp/acdc_run_150_earlystop.log"),
    "10th (attention r=2)": Path("/tmp/acdc_run_150_resattn.log"),
    "11th (attention r=1)": Path("/tmp/acdc_run_150_resattn_r1.log"),
    "12th (no attention)": Path("/tmp/acdc_run_150_resunet_noattn.log"),
}

#: Where each iteration's evaluation_report.json was copied for
#: delivery to the user (see the "Copy the final evaluation report to
#: /tmp/deliverables/" step every iteration's writeup follows).
_EVAL_REPORTS = {
    "8th": Path("/tmp/deliverables/acdc_iteration8_earlystop_evaluation_report.json"),
    "9th": Path("/tmp/deliverables/acdc_iteration9_cosine_annealing_evaluation_report.json"),
    "10th": Path("/tmp/deliverables/acdc_iteration10_res_attention_unet_evaluation_report.json"),
    "11th": Path("/tmp/deliverables/acdc_iteration11_wide_gate_bottleneck_evaluation_report.json"),
    "12th": Path(
        "/tmp/deliverables/acdc_iteration12_plain_res_unet_no_attention_evaluation_report.json"
    ),
}

#: The twelfth iteration's full pipeline output directory -- still on
#: disk this session, has padded_images/padded_labels/predictions for
#: every case plus manifest.json listing the 60 test cases.
_ITERATION_12_OUTPUT_DIR = Path("/tmp/acdc_validation_out_150_resunet_noattn")

#: The eighth (reference-baseline) iteration's full pipeline output
#: directory -- same layout as the twelfth's, also still on disk. Every
#: iteration uses the same 150-patient list and the same
#: `_patient_level_split(..., seed=42)` call, so its 60-case test set
#: is identical to the twelfth iteration's -- confirmed by comparing
#: both manifests' `test` image filenames before relying on this.
_ITERATION_8_OUTPUT_DIR = Path("/tmp/acdc_validation_out_150_earlystop")

#: Test cases used for the prediction-vs-ground-truth comparison plots:
#: the two weakest twelfth-iteration cases (see docs/
#: real_data_validation.md's "Twelfth iteration" section) plus one
#: representative case near the test set's median Dice. Same cases are
#: used for the eighth iteration's comparison plots, since the test
#: set is identical -- lets the two be compared frame for frame.
_COMPARISON_CASES = ["patient142_frame12", "patient086_frame08", "patient001_frame01"]

#: Matches this project's training-log line format, e.g. "Epoch 19/50
#: - train loss: 0.1666" / "Epoch 19/50 - val Dice: 0.7906" -- see
#: miai_segmentation.three_d.train's INFO logging (there is no CSV
#: log, only these text lines, which is why this parses them at all).
_EPOCH_LOG_PATTERN = re.compile(
    r"Epoch (?P<epoch>\d+)/\d+ - (?P<metric>train loss|val Dice): (?P<value>[\d.]+)"
)


def parse_training_log(log_path: Path) -> list[dict[str, float]]:
    """Parse per-epoch ``train_loss``/``val_dice`` rows out of a text training log.

    Args:
        log_path: Path to a ``validate_acdc.py`` run's stdout/stderr
            log (redirected there by the ``nohup ... > log 2>&1``
            pattern this project's background runs use).

    Returns:
        One dict per epoch that has both a train-loss and a val-Dice
        line (``{"epoch": ..., "train_loss": ..., "val_dice": ...}``),
        sorted by epoch. An epoch cut off mid-write (e.g. by an
        interrupted run) contributes no row rather than a partial one.
    """
    rows: dict[int, dict[str, float]] = {}
    for line in log_path.read_text().splitlines():
        match = _EPOCH_LOG_PATTERN.search(line)
        if not match:
            continue
        epoch = int(match.group("epoch"))
        key = "train_loss" if match.group("metric") == "train loss" else "val_dice"
        rows.setdefault(epoch, {})[key] = float(match.group("value"))
    return [
        {"epoch": float(epoch), **values}
        for epoch, values in sorted(rows.items())
        if "train_loss" in values and "val_dice" in values
    ]


def _write_csv(rows: list[dict[str, float]], fieldnames: list[str], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_all_training_curves(output_dir: Path) -> list[Path]:
    """Plot the twelfth iteration's own curve, and a 4-way val-Dice comparison.

    Args:
        output_dir: Directory PNGs and the intermediate CSVs are
            written to.

    Returns:
        Paths to the PNGs written, in the order described in the
        module docstring's item 1.
    """
    written: list[Path] = []
    parsed = {name: parse_training_log(path) for name, path in _TRAINING_LOGS.items()}

    twelfth_rows = parsed["12th (no attention)"]
    twelfth_csv = output_dir / "iteration12_training_log.csv"
    _write_csv(twelfth_rows, ["epoch", "train_loss", "val_dice"], twelfth_csv)
    written.append(
        plot_training_curves(
            str(twelfth_csv),
            str(output_dir / "iteration12_train_loss_and_val_dice.png"),
            PlotTrainingCurvesConfig(
                title="Twelfth iteration (no attention): train loss and val Dice per epoch",
                ylabel="value",
            ),
        )
    )

    # Combined val-Dice comparison, forward-filled past each run's own
    # early-stopping point so every line stays flat (not truncated)
    # once that iteration stopped -- plot_training_curves requires a
    # numeric value in every row for every requested column.
    max_epoch = max(int(rows[-1]["epoch"]) for rows in parsed.values())
    comparison_fieldnames = ["epoch", *parsed.keys()]
    comparison_rows: list[dict[str, float]] = []
    last_seen = dict.fromkeys(parsed, 0.0)
    by_epoch = {
        name: {int(row["epoch"]): row["val_dice"] for row in rows} for name, rows in parsed.items()
    }
    for epoch in range(1, max_epoch + 1):
        row: dict[str, float] = {"epoch": float(epoch)}
        for name in parsed:
            value = by_epoch[name].get(epoch, last_seen[name])
            last_seen[name] = value
            row[name] = value
        comparison_rows.append(row)
    comparison_csv = output_dir / "val_dice_comparison.csv"
    _write_csv(comparison_rows, comparison_fieldnames, comparison_csv)
    written.append(
        plot_training_curves(
            str(comparison_csv),
            str(output_dir / "val_dice_comparison_8th_10th_11th_12th.png"),
            PlotTrainingCurvesConfig(
                title="Validation Dice per epoch: baseline vs. the three attention-lever "
                "iterations (flat after each run's own early stopping)",
                ylabel="validation Dice",
            ),
        )
    )
    return written


def plot_all_case_comparisons(output_dir: Path, iteration_dir: Path, label: str) -> list[Path]:
    """Plot ground-truth-vs-prediction comparisons for the fixed comparison cases.

    Args:
        output_dir: Directory PNGs are written to.
        iteration_dir: A completed run's full pipeline output directory
            (e.g. ``_ITERATION_12_OUTPUT_DIR``), providing
            ``padded_images``/``padded_labels``/``predictions``.
        label: Short tag identifying the iteration, used in the plot
            titles (e.g. ``"8th (baseline)"``) so two iterations' PNGs
            written to the same case are visually distinguishable.

    Returns:
        Two PNG paths per case in ``_COMPARISON_CASES`` (a mask
        comparison and an MRI+prediction overlay), in that order.
    """
    written: list[Path] = []
    for case in _COMPARISON_CASES:
        image_path = iteration_dir / "padded_images" / f"{case}_preprocessed.nii.gz"
        label_path = iteration_dir / "padded_labels" / f"{case}_gt_preprocessed.nii.gz"
        pred_path = iteration_dir / "predictions" / f"{case}_preprocessed_pred.nii.gz"

        image = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))
        gt_label = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
        prediction = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path)))

        written.append(
            plot_comparison(
                {"Ground truth": gt_label, "Prediction": prediction},
                str(output_dir / f"{case}_gt_vs_prediction.png"),
                PlotComparisonConfig(cmap="viridis", include_difference_map=True),
            )
        )
        written.append(
            plot_slice(
                image,
                str(output_dir / f"{case}_mri_with_prediction_overlay.png"),
                PlotSliceConfig(
                    mask_cmap="Reds",
                    mask_alpha=0.45,
                    title=f"{case} ({label}): MRI + predicted mask",
                ),
                mask=prediction,
            )
        )
    return written


def _select_points_of_interest(
    per_case: list[dict[str, object]],
) -> list[tuple[str, dict[str, object]]]:
    """Pick the best-, median-, and worst-Dice cases from a report's ``per_case`` list.

    Args:
        per_case: ``evaluation_report.json["per_case"]`` -- one dict
            per test case, each with a ``"dice"`` (macro) key.

    Returns:
        Three ``(row_label, case)`` pairs, in best/median/worst order.
        "Median" is the case whose own Dice is closest to the set's
        statistical median, not necessarily the middle element by
        sort order (ties broken by whichever sorts first).
    """
    dice_values = [float(case["dice"]) for case in per_case]  # type: ignore[arg-type]
    median_dice = statistics.median(dice_values)
    best = max(per_case, key=lambda case: case["dice"])  # type: ignore[arg-type,return-value]
    worst = min(per_case, key=lambda case: case["dice"])  # type: ignore[arg-type,return-value]
    median_case = min(per_case, key=lambda case: abs(case["dice"] - median_dice))  # type: ignore[operator]
    return [
        (f"Best (Dice={best['dice']:.3f})", best),
        (f"Median (Dice={median_case['dice']:.3f})", median_case),
        (f"Worst (Dice={worst['dice']:.3f})", worst),
    ]


def plot_points_of_interest_grid(
    report_path: Path, iteration_dir: Path, output_path: Path, label: str
) -> Path:
    """Plot a 3x3 grid: best/median/worst-Dice cases x ground truth/prediction/diff.

    One PNG, nine panels: each row is a representative test case
    picked by macro Dice (see :func:`_select_points_of_interest`), each
    column is one segmentation view of that case's middle slice
    (ground truth label map, predicted label map, their absolute
    difference) -- a fast way to spot where a model does best, does
    typically, and fails worst, without opening nine separate files.

    Args:
        report_path: An iteration's ``evaluation_report.json``.
        iteration_dir: That same iteration's full pipeline output
            directory, providing ``padded_labels``/``predictions``.
        output_path: Where the PNG is written. Parent directories are
            created if missing.
        label: Short tag identifying the iteration, used in the
            figure's overall title (e.g. ``"8th (baseline)"``).

    Returns:
        ``output_path`` as a :class:`pathlib.Path`.
    """
    report = json.loads(report_path.read_text())
    rows = _select_points_of_interest(report["per_case"])
    columns = ["Ground truth", "Prediction", "|Prediction - Ground truth|"]

    fig, axes = plt.subplots(3, 3, figsize=(10.5, 11))
    for row_idx, (row_label, case) in enumerate(rows):
        case_root = str(case["case"]).removesuffix("_preprocessed_pred.nii.gz")
        label_path = iteration_dir / "padded_labels" / f"{case_root}_gt_preprocessed.nii.gz"
        pred_path = iteration_dir / "predictions" / str(case["case"])

        gt_volume = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
        pred_volume = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path)))
        mid = gt_volume.shape[0] // 2
        gt_slice = gt_volume[mid]
        pred_slice = pred_volume[mid]
        diff_slice = np.abs(pred_slice.astype(np.float64) - gt_slice.astype(np.float64))

        panels = [gt_slice, pred_slice, diff_slice]
        cmaps = ["viridis", "viridis", "inferno"]
        for col_idx, (panel, cmap) in enumerate(zip(panels, cmaps, strict=True)):
            ax = axes[row_idx, col_idx]
            ax.imshow(panel, cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(columns[col_idx])
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\n{case_root}", fontsize=9)

    fig.suptitle(f"{label}: points of interest (best / median / worst test-case Dice)")
    fig.tight_layout()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_macro_dice_bar_chart(output_dir: Path) -> Path:
    """Plot a cross-iteration macro test Dice bar chart (8th-12th).

    Args:
        output_dir: Directory the PNG is written to.

    Returns:
        The PNG path written.
    """
    macro_dice = {}
    for iteration, report_path in _EVAL_REPORTS.items():
        report = json.loads(report_path.read_text())
        macro_dice[iteration] = report["mean"]["dice"]
    return plot_metric_summary(
        macro_dice,
        str(output_dir / "macro_test_dice_by_iteration.png"),
        PlotMetricSummaryConfig(
            kind="bar",
            title="Macro test Dice by iteration (8th-12th)",
            ylabel="macro Dice (foreground only)",
        ),
    )


def plot_per_class_dice_box(report_path: Path, output_dir: Path, label: str) -> Path:
    """Plot a per-case, per-class (RV/Myo/LV) test Dice box plot for one iteration.

    Args:
        report_path: An iteration's ``evaluation_report.json``.
        output_dir: Directory the PNG is written to.
        label: Short tag identifying the iteration, used in the plot
            title and output filename (e.g. ``"8th (baseline)"``).

    Returns:
        The PNG path written.
    """
    report = json.loads(report_path.read_text())
    per_class_dice = {
        "RV": [case["dice_class_1"] for case in report["per_case"]],
        "Myo": [case["dice_class_2"] for case in report["per_case"]],
        "LV": [case["dice_class_3"] for case in report["per_case"]],
    }
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return plot_metric_summary(
        per_class_dice,
        str(output_dir / f"{slug}_per_class_dice_distribution.png"),
        PlotMetricSummaryConfig(
            kind="box",
            title=f"{label}: per-case test Dice by class (60 test cases)",
            ylabel="Dice",
        ),
    )


def run_qc_montages(output_dir: Path, iteration_dir: Path) -> list[Path]:
    """Run VisualizationStage over every one of an iteration's test-set images.

    The same class ``examples/validate_acdc.py --visualize`` now runs
    at the end of a live pipeline run -- this function exercises it
    against a completed run's already-existing output instead of
    waiting for a new one.

    Args:
        output_dir: Directory QC montage PNGs are written to.
        iteration_dir: A completed run's full pipeline output
            directory (e.g. ``_ITERATION_12_OUTPUT_DIR``), providing
            ``manifest.json``.

    Returns:
        One path per test-set case (60 for both the eighth and
        twelfth iterations, since their test sets are identical).
    """
    manifest = json.loads((iteration_dir / "manifest.json").read_text())
    test_image_paths = [Path(case["image"]) for case in manifest["test"]]

    context = PipelineContext()
    context.set("preprocessed_paths", test_image_paths)
    stage = VisualizationStage(VisualizationStageConfig(output_dir=str(output_dir)))
    context = stage.run(context)
    result: list[Path] = context.require("qc_visualization_paths")
    return result


def _zip_files(paths: list[Path], zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, arcname=path.name)
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/acdc_visualizations"))
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    curves_dir = output_dir / "curves"
    summaries_dir = output_dir / "metric_summaries"

    curve_paths = plot_all_training_curves(curves_dir)
    macro_bar_path = plot_macro_dice_bar_chart(summaries_dir)

    iterations = [
        ("12th (no attention)", _ITERATION_12_OUTPUT_DIR, _EVAL_REPORTS["12th"], "iteration12"),
        ("8th (baseline)", _ITERATION_8_OUTPUT_DIR, _EVAL_REPORTS["8th"], "iteration8"),
    ]

    case_paths: list[Path] = []
    box_paths: list[Path] = []
    qc_zips: list[Path] = []
    poi_paths: list[Path] = []
    for label, iteration_dir, report_path, slug in iterations:
        cases_dir = output_dir / "case_comparisons" / slug
        case_paths += plot_all_case_comparisons(cases_dir, iteration_dir, label)
        box_paths.append(plot_per_class_dice_box(report_path, summaries_dir, label))
        poi_paths.append(
            plot_points_of_interest_grid(
                report_path, iteration_dir, output_dir / f"{slug}_points_of_interest.png", label
            )
        )

        qc_dir = output_dir / "qc_montages" / slug
        qc_paths = run_qc_montages(qc_dir, iteration_dir)
        qc_zips.append(_zip_files(qc_paths, output_dir / f"{slug}_qc_montages.zip"))

    print()
    print("=== ACDC results visualization finished ===")
    print(f"Training curves: {len(curve_paths)} PNGs under {curves_dir}")
    print(f"Macro Dice bar chart: {macro_bar_path}")
    print(f"Case comparisons: {len(case_paths)} PNGs under {output_dir / 'case_comparisons'}")
    print(f"Per-class Dice box plots: {box_paths}")
    print(f"Points-of-interest grids: {poi_paths}")
    print(f"QC montages zipped to: {qc_zips}")


if __name__ == "__main__":
    main()
