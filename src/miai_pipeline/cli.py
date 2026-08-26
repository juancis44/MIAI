"""Command-line interface for miai-pipeline.

Lets a pipeline defined in a YAML config be validated, inspected, or run
without writing any Python -- consistent with MIAI's "configuration over
code" principle (see docs/api_design.md and docs/coding_standards.md).
Installed as the ``miai-pipeline`` console script (see ``[project.scripts]``
in the root ``pyproject.toml``).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from miai_core.exceptions import MIAIError
from miai_core.logging import configure_logging, get_logger
from miai_pipeline.config import PipelineConfig
from miai_pipeline.context import PipelineContext
from miai_pipeline.pipeline import Pipeline

logger = get_logger(__name__)


def _parse_set_value(raw: str) -> tuple[str, str]:
    """Parse a ``--set KEY=VALUE`` argument into a ``(key, value)`` pair."""
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--set expects KEY=VALUE, got {raw!r} (no '=' found).")
    key, value = raw.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError(f"--set expects a non-empty KEY in {raw!r}.")
    return key, value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="miai-pipeline", description="Run and inspect MIAI pipelines defined in YAML."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level for stage progress messages (default: INFO).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a pipeline from a YAML config.")
    run_parser.add_argument("config_path", help="Path to a pipeline YAML config.")
    run_parser.add_argument(
        "--set",
        dest="set_values",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        type=_parse_set_value,
        help="Set an initial context value before running (repeatable), "
        "e.g. --set dicom_dir=data/raw_dicom.",
    )

    validate_parser = subparsers.add_parser(
        "validate", help="Validate a pipeline YAML config without running it."
    )
    validate_parser.add_argument("config_path", help="Path to a pipeline YAML config.")

    subparsers.add_parser("list-stages", help="List every registered pipeline stage type.")

    return parser


def _build_pipeline_from_path(config_path: str) -> Pipeline:
    config = PipelineConfig.from_yaml(config_path)
    return Pipeline.from_config(config)


def _run(args: argparse.Namespace) -> int:
    pipeline = _build_pipeline_from_path(args.config_path)

    context = PipelineContext()
    for key, value in args.set_values:
        context.set(key, value)

    result = pipeline.run(context)
    logger.info("Pipeline finished. Final context keys: %s", ", ".join(sorted(result.keys())))
    return 0


def _validate(args: argparse.Namespace) -> int:
    pipeline = _build_pipeline_from_path(args.config_path)
    stage_names = ", ".join(stage.name for stage in pipeline.stages)
    print(f"Config is valid: {len(pipeline.stages)} stage(s) -> {stage_names}")
    return 0


def _list_stages(_args: argparse.Namespace) -> int:
    from miai_pipeline.stages import STAGE_REGISTRY

    for name in sorted(STAGE_REGISTRY):
        print(name)
    return 0


_HANDLERS = {"run": _run, "validate": _validate, "list-stages": _list_stages}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``miai-pipeline`` console script.

    Args:
        argv: Command-line arguments, excluding the program name.
            Defaults to ``sys.argv[1:]`` if ``None``.

    Returns:
        Process exit code: ``0`` on success, ``1`` if building or
        running the pipeline raised a MIAI exception (the exception is
        logged rather than raised, so the CLI exits cleanly instead of
        printing a Python traceback).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(level=args.log_level.upper(), force=True)

    try:
        return _HANDLERS[args.command](args)
    except MIAIError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    # Untestable by definition: pytest imports this module rather than
    # running it as __main__, so this guard can never execute under
    # the test suite regardless of how main() itself is exercised.
    sys.exit(main())
