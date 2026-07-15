"""Exporting a trained model to a portable inference format.

Rather than serving models live, MIAI's reference deployment task is
*portable export*: converting a trained :class:`torch.nn.Module` (plus
its checkpoint) into a self-contained artifact -- a TorchScript module
or an ONNX graph -- that can be loaded and run without the original
Python model-construction code, e.g. from a C++ runtime, ONNX Runtime,
or a separate inference-serving process not built on MIAI at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from miai_core.config import MIAIBaseConfig
from miai_core.logging import get_logger
from miai_deploy.exceptions import DeployError

logger = get_logger(__name__)


class ExportConfig(MIAIBaseConfig):
    """Configuration for :func:`export_model`.

    Attributes:
        format: ``"torchscript"`` (via ``torch.jit.trace``, no extra
            dependency beyond ``torch`` itself) or ``"onnx"`` (via
            ``torch.onnx.export``, requires the ``onnx`` package).
        example_input_shape: Shape of the dummy tensor used to trace
            the model, ``(B, C, D, H, W)``. Must be a shape the model
            actually accepts -- for MIAI's reference
            :class:`~miai_segmentation.models.UNetConfig` architecture,
            each spatial dimension must be divisible by
            ``2 ** (len(channels) - 1)`` (the number of downsampling
            levels), matching the same constraint documented on
            :class:`~miai_diffusion.model.DiffusionUNetConfig`.
        opset_version: ONNX opset version. Ignored for
            ``format="torchscript"``.
        device: ``"cpu"`` or ``"cuda"``. Exported artifacts are not
            tied to this device at load time, but tracing/exporting
            itself runs on it.
    """

    format: Literal["torchscript", "onnx"] = "torchscript"
    example_input_shape: tuple[int, ...] = (1, 1, 32, 32, 32)
    opset_version: int = 17
    device: str = "cpu"


def export_model(
    model: torch.nn.Module,
    checkpoint_path: str,
    config: ExportConfig,
    output_path: str,
) -> Path:
    """Load a checkpoint into ``model`` and export it to ``output_path``.

    Args:
        model: An untrained model with the same architecture used to
            produce ``checkpoint_path``.
        checkpoint_path: Path to a state dict saved by e.g.
            :func:`miai_segmentation.train.train_model`.
        config: Export parameters.
        output_path: Where the exported artifact is written. Parent
            directories are created if missing.

    Returns:
        ``output_path`` as a :class:`pathlib.Path`, for chaining.

    Raises:
        DeployError: If ``config.format`` is not a recognized value.
    """
    device = torch.device(config.device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    example_input = torch.zeros(config.example_input_shape, device=device)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if config.format == "torchscript":
        with torch.no_grad():
            traced = torch.jit.trace(model, example_input)
        traced.save(str(out_path))
    elif config.format == "onnx":
        with torch.no_grad():
            torch.onnx.export(
                model,
                example_input,
                str(out_path),
                opset_version=config.opset_version,
                input_names=["input"],
                output_names=["output"],
            )
    else:
        raise DeployError(f"Unknown export format: {config.format!r}")

    logger.info("Exported %s model from %s to %s", config.format, checkpoint_path, out_path)
    return out_path
