"""Patient-level k-fold cross-validation and grouped-LOPO for the ACDC validation effort.

Every iteration in ``examples/validate_acdc.py`` (through the
sixteenth) reports a single number from a single, fixed, arbitrarily-
seeded patient-level train/val/test split. The sixteenth iteration's
own result (macro test Dice 0.7367 on a 120/15/15 split, versus the
eighth iteration's 0.7740 on 90/30/30) is itself evidence that this is
a noisy estimator: swapping which 15-30 patients happen to land in the
held-out test set can move the headline number by several points, on
its own, with no real change to the model. See
``docs/real_data_validation.md``'s "Sixteenth iteration" section for
the full argument.

This script answers that by rotating which patients are held out,
rather than fixing it once:

- **k-fold cross-validation** (``--mode kfold``, default ``--k 5``):
  the full 150-patient set is shuffled once (fixed ``--seed``) and cut
  into ``k`` equal-sized folds. Each fold takes a turn as the test
  set; the other ``k - 1`` folds' patients are pooled and split again
  into train/val (``--val-patients``, default 15, taken off the front
  of the pooled patients in their already-shuffled order -- see
  ``k_fold_splits``'s docstring for why this needs no second random
  seed). With ``k=5`` and 150 patients, every patient is tested
  exactly once across the 5 runs, and every run trains on ~105
  patients (105/15/30) with the eighth iteration's exact baseline
  architecture -- the only thing that changes fold to fold is which
  patients land where.

- **Grouped leave-one-out** (``--mode lopo-groups``, default
  ``--group-size 10``): the same idea taken further -- true
  leave-one-patient-out over 150 patients would mean 150 full training
  runs, infeasible given this project's own measured ~5h/run and
  documented sandbox instability (see ``docs/real_data_validation.md``
  and the project's sandbox-workflow notes). Grouping patients into
  blocks of ``--group-size`` (default 10, giving 15 groups over 150
  patients) is a middle ground: finer-grained than 5-fold (15 distinct
  held-out sets instead of 5, so more resolution on how much the
  result varies by which patients are held out), but still only 15
  full runs, not 150.

**Why this reuses, not reimplements, the pipeline.**
``examples/validate_acdc.py``'s ``run_validation()`` gained an
optional ``patient_split`` parameter (a ``{"train", "val", "test"}``
mapping of explicit patient-ID sets) specifically to support this
script -- passing it bypasses the fraction-based
``_patient_level_split`` in favor of
``_patient_level_split_from_assignment``, but every other stage
(preprocessing, padding, training, inference, evaluation) is the exact
same code, run once per fold. No duplicated pipeline logic here; this
module is purely fold *generation*, per-fold *orchestration*, and
cross-fold *aggregation*.

**Why a pooled per-case statistic matters more than any single fold's
number.** Because every patient is held out exactly once (in both
modes), concatenating every fold's ``per_case`` test results gives a
Dice/Hausdorff/etc. value for all 150 patients, not just whichever 15
or 30 happened to land in one iteration's fixed test set. The pooled
mean/stdev across all 150 patients -- reported alongside the
per-fold mean/stdev -- is the more statistically meaningful number
this whole effort has been missing.

**Disk management.** Each fold's preprocessed/padded image directories
are the bulk of a fold's output (~1.3GB, see ``docs/real_data_
validation.md``'s "Sixteenth iteration" disk-usage notes) and are pure
intermediates -- safely regenerable from ``--data-dir`` and not needed
once a fold's ``evaluation_report.json``/checkpoint exist. By default
(``--cleanup-images``, on unless ``--keep-images`` is passed) this
script deletes each fold's ``preprocessed_images/``, ``padded_images/``,
``preprocessed_labels/``, ``padded_labels/``, and ``labels/``
directories immediately after that fold's evaluation completes,
keeping only ``checkpoints/`` (~1.6MB), ``manifest.json``,
``evaluation_report.json``, and ``predictions/`` (a few hundred KB) --
around 2MB/fold instead of 1.3GB/fold, so a full 5-fold or 15-group
study doesn't exhaust the sandbox's disk the way running all folds
with intermediates retained would.

**Resuming across sessions.** Given this project's documented sandbox
instability (a background training run can die mid-run, sometimes from
the sandbox container itself restarting -- see ``docs/real_data_
validation.md``'s "Fifteenth iteration" section), a 5-run or 15-run
study is not going to complete in one sitting reliably. ``--start-fold``/
``--end-fold`` (1-indexed, inclusive) run only a subrange of folds, so
a failed run can be relaunched starting from the fold that died without
repeating already-completed folds. ``--aggregate-only`` skips training
entirely and rebuilds the cross-fold summary from whatever
``evaluation_report.json`` files already exist under ``--output-root``
-- safe to run at any point to check progress, and is how the final
report gets built once every fold is done.

Run (5-fold, default):
    python examples/cross_validate_acdc.py --data-dir /path/to/ACDC \\
        --output-root examples/output/acdc_cv5 --mode kfold --k 5

Run (grouped LOPO, groups of 10):
    python examples/cross_validate_acdc.py --data-dir /path/to/ACDC \\
        --output-root examples/output/acdc_lopo10 --mode lopo-groups --group-size 10

Resume folds 3-5 after an earlier interrupted run:
    python examples/cross_validate_acdc.py --data-dir /path/to/ACDC \\
        --output-root examples/output/acdc_cv5 --mode kfold --k 5 \\
        --start-fold 3 --end-fold 5

Check/report progress without training anything:
    python examples/cross_validate_acdc.py --output-root examples/output/acdc_cv5 \\
        --aggregate-only
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
from pathlib import Path

from validate_acdc import DEFAULT_PATIENTS, run_validation

from miai_core.logging import configure_logging, get_logger

logger = get_logger(__name__)

#: Fixed inner-validation patient count for every fold, in every mode.
#: Matches the sixteenth iteration's own val set size (15 patients),
#: so a fold's train/val/test proportions are directly comparable to
#: that iteration's single fixed split, not just to each other. See
#: ``k_fold_splits``'s docstring for how these patients are chosen.
_DEFAULT_VAL_PATIENTS = 15

#: Intermediate output subdirectories safe to delete once a fold's
#: evaluation is done -- see the module docstring's "Disk management"
#: section. Everything else under a fold's output dir (checkpoints/,
#: manifest.json, evaluation_report.json, predictions/) is kept.
_CLEANUP_SUBDIRS = (
    "preprocessed_images",
    "padded_images",
    "preprocessed_labels",
    "padded_labels",
    "labels",
)


def k_fold_splits(
    patients: list[str], k: int, seed: int, val_patients: int
) -> list[dict[str, object]]:
    """Build ``k`` patient-level folds, each a turn as the held-out test set.

    ``patients`` is shuffled once with ``seed``, then cut into ``k``
    contiguous, near-equal-sized slices (sizes differ by at most 1 if
    ``len(patients)`` doesn't divide evenly by ``k``) -- each slice is
    one fold's test set. For a given fold, "remaining" is every other
    patient in their *original shuffled order* with that fold's test
    patients removed; the first ``val_patients`` of remaining become
    that fold's validation set, and the rest become its training set.
    Deliberately no second random draw for val/train: since the whole
    list was already shuffled once, taking a prefix of an already-
    randomized order is itself a valid random sample, and doing it
    this way means every fold's val/train assignment is fully
    determined by the one ``seed`` -- reproducible without needing to
    track a per-fold seed too. Note this means val sets can overlap
    somewhat across folds (a patient not selected as fold i's test
    patient is eligible to be in several folds' val sets) -- harmless,
    since val here only selects the early-stopping checkpoint and
    never contributes to a reported test metric.

    Returns one dict per fold: ``{"fold_id": str, "train": set[str],
    "val": set[str], "test": set[str]}``.
    """
    shuffled = list(patients)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    boundaries = [round(i * n / k) for i in range(k + 1)]

    folds: list[dict[str, object]] = []
    for i in range(k):
        test = set(shuffled[boundaries[i] : boundaries[i + 1]])
        remaining = [p for p in shuffled if p not in test]
        val = set(remaining[:val_patients])
        train = set(remaining[val_patients:])
        folds.append({"fold_id": f"fold{i + 1}", "train": train, "val": val, "test": test})
    return folds


def grouped_lopo_splits(
    patients: list[str], group_size: int, seed: int, val_patients: int
) -> list[dict[str, object]]:
    """Build one fold per contiguous group of ``group_size`` patients.

    Same shuffle-once, prefix-of-remaining-as-val logic as
    :func:`k_fold_splits` -- see its docstring -- except groups are
    fixed-size (``group_size`` patients each) rather than ``k``
    equal-sized folds, so the number of folds is
    ``ceil(len(patients) / group_size)``, not chosen directly. With
    the default ``group_size=10`` over 150 patients this gives exactly
    15 groups of 10; a ``len(patients)`` not evenly divisible by
    ``group_size`` leaves a final, smaller group rather than erroring.

    Returns one dict per group, same shape as :func:`k_fold_splits`,
    with ``"fold_id"`` values like ``"group1"``, ``"group2"``, ...
    """
    shuffled = list(patients)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)

    folds: list[dict[str, object]] = []
    for i, start in enumerate(range(0, n, group_size)):
        test = set(shuffled[start : start + group_size])
        remaining = [p for p in shuffled if p not in test]
        val = set(remaining[:val_patients])
        train = set(remaining[val_patients:])
        folds.append({"fold_id": f"group{i + 1}", "train": train, "val": val, "test": test})
    return folds


def _cleanup_fold_intermediates(fold_dir: Path) -> None:
    """Delete a completed fold's large, regenerable preprocessing intermediates.

    See the module docstring's "Disk management" section for what
    stays (checkpoints, manifest, evaluation report, predictions) and
    why: a fold's preprocessed/padded images are ~1.3GB and add
    nothing once that fold's checkpoint and metrics already exist.
    """
    for name in _CLEANUP_SUBDIRS:
        shutil.rmtree(fold_dir / name, ignore_errors=True)


def run_fold(
    data_dir: Path,
    output_root: Path,
    fold: dict[str, object],
    max_epochs: int,
    visualize: bool,
    cleanup_images: bool,
) -> dict[str, object]:
    """Run the full pipeline for one fold and return its ``run_validation`` summary."""
    fold_id = fold["fold_id"]
    fold_dir = output_root / str(fold_id)
    logger.info(
        "=== Starting %s: %d train / %d val / %d test patients ===",
        fold_id,
        len(fold["train"]),
        len(fold["val"]),
        len(fold["test"]),
    )
    summary = run_validation(
        data_dir,
        fold_dir,
        max_epochs,
        visualize=visualize,
        patient_split={"train": fold["train"], "val": fold["val"], "test": fold["test"]},
    )
    logger.info(
        "=== Finished %s: macro test Dice %.4f ===",
        fold_id,
        summary["mean_metrics"]["dice"],
    )
    if cleanup_images:
        _cleanup_fold_intermediates(fold_dir)
        logger.info("Cleaned up %s's preprocessing intermediates", fold_id)
    return summary


def aggregate_folds(output_root: Path) -> dict[str, object]:
    """Rebuild the cross-fold summary from every fold's ``evaluation_report.json`` on disk.

    Scans ``output_root``'s immediate subdirectories for a
    ``evaluation_report.json`` (written by
    :class:`~miai_pipeline.stages.evaluation.EvaluationStage`) plus a
    ``manifest.json`` (to recover each fold's test-patient count), and
    computes both per-fold and pooled-across-all-patients statistics --
    see the module docstring's "Why a pooled per-case statistic
    matters" section for what the pooled numbers mean and why they are
    the more meaningful headline result. Safe to call at any point,
    including with only some folds finished (reports on whatever
    exists), which is what makes ``--aggregate-only`` useful as a
    progress check mid-study.
    """
    fold_dirs = sorted(p for p in output_root.iterdir() if p.is_dir())
    per_fold: list[dict[str, object]] = []
    pooled_per_case: list[dict[str, object]] = []

    for fold_dir in fold_dirs:
        report_path = fold_dir / "evaluation_report.json"
        if not report_path.exists():
            logger.warning("Skipping %s: no evaluation_report.json yet", fold_dir.name)
            continue
        report = json.loads(report_path.read_text())
        manifest_path = fold_dir / "manifest.json"
        test_patient_count = None
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            test_patient_count = len(
                {Path(c["image"]).name.split("_frame")[0] for c in manifest["test"]}
            )
        per_fold.append(
            {
                "fold_id": fold_dir.name,
                "test_patients": test_patient_count,
                "test_cases": len(report["per_case"]),
                "mean_metrics": report["mean"],
            }
        )
        pooled_per_case.extend(report["per_case"])

    if not per_fold:
        return {"folds": [], "n_folds_completed": 0}

    fold_dices = [f["mean_metrics"]["dice"] for f in per_fold]
    pooled_dices = [c["dice"] for c in pooled_per_case]

    summary: dict[str, object] = {
        "n_folds_completed": len(per_fold),
        "folds": per_fold,
        "across_folds": {
            "macro_test_dice_mean": statistics.mean(fold_dices),
            "macro_test_dice_stdev": (statistics.stdev(fold_dices) if len(fold_dices) > 1 else 0.0),
            "macro_test_dice_min": min(fold_dices),
            "macro_test_dice_max": max(fold_dices),
        },
        "pooled_per_case": {
            "n_cases": len(pooled_dices),
            "dice_mean": statistics.mean(pooled_dices),
            "dice_stdev": (statistics.stdev(pooled_dices) if len(pooled_dices) > 1 else 0.0),
            "dice_median": statistics.median(pooled_dices),
            "dice_min": min(pooled_dices),
            "dice_max": max(pooled_dices),
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, help="ACDC root dir with patientXXX/ subfolders")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Parent dir; one subdir per fold is created under it",
    )
    parser.add_argument("--mode", choices=["kfold", "lopo-groups"], default="kfold")
    parser.add_argument("--k", type=int, default=5, help="Number of folds for --mode kfold")
    parser.add_argument(
        "--group-size",
        type=int,
        default=10,
        help="Patients per held-out group for --mode lopo-groups",
    )
    parser.add_argument("--val-patients", type=int, default=_DEFAULT_VAL_PATIENTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--keep-images",
        action="store_true",
        help="Do not delete a fold's preprocessed/padded image directories after it "
        "completes (default: delete them, see the module docstring's 'Disk "
        "management' section). Pass this only if disk space is not a concern.",
    )
    parser.add_argument(
        "--start-fold",
        type=int,
        default=1,
        help="1-indexed, inclusive -- resume a partially-run study",
    )
    parser.add_argument(
        "--end-fold", type=int, default=None, help="1-indexed, inclusive; default: last fold"
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip training entirely; just rebuild and print the cross-fold summary "
        "from whatever evaluation_report.json files already exist under --output-root.",
    )
    args = parser.parse_args()

    configure_logging(level="INFO", force=True)

    if args.mode == "kfold":
        folds = k_fold_splits(DEFAULT_PATIENTS, args.k, args.seed, args.val_patients)
    else:
        folds = grouped_lopo_splits(DEFAULT_PATIENTS, args.group_size, args.seed, args.val_patients)

    if not args.aggregate_only:
        if args.data_dir is None:
            parser.error("--data-dir is required unless --aggregate-only is passed")
        end_fold = args.end_fold if args.end_fold is not None else len(folds)
        args.output_root.mkdir(parents=True, exist_ok=True)
        for i, fold in enumerate(folds, start=1):
            if i < args.start_fold or i > end_fold:
                continue
            run_fold(
                args.data_dir,
                args.output_root,
                fold,
                args.max_epochs,
                args.visualize,
                cleanup_images=not args.keep_images,
            )

    summary = aggregate_folds(args.output_root)
    summary_path = args.output_root / "cv_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    n_done = summary["n_folds_completed"]
    print()
    print(f"=== Cross-validation summary ({args.mode}, {n_done}/{len(folds)} folds) ===")
    if n_done:
        for f in summary["folds"]:
            dice = f["mean_metrics"]["dice"]
            print(f"  {f['fold_id']}: {f['test_patients']} test patients, macro Dice {dice:.4f}")
        af = summary["across_folds"]
        mean, stdev = af["macro_test_dice_mean"], af["macro_test_dice_stdev"]
        lo, hi = af["macro_test_dice_min"], af["macro_test_dice_max"]
        print(f"Across folds: mean {mean:.4f} +/- {stdev:.4f} (min {lo:.4f}, max {hi:.4f})")
        pc = summary["pooled_per_case"]
        print(
            f"Pooled per-case ({pc['n_cases']} cases across all held-out patients): "
            f"mean {pc['dice_mean']:.4f} +/- {pc['dice_stdev']:.4f}, "
            f"median {pc['dice_median']:.4f}, range [{pc['dice_min']:.4f}, {pc['dice_max']:.4f}]"
        )
    print(f"Full summary written to {summary_path}")


if __name__ == "__main__":
    main()
