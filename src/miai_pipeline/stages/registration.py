"""Registration stage: aligns each case to a common fixed reference image."""

from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk

from miai_core.config import MIAIBaseConfig
from miai_core.io import ensure_dir
from miai_core.logging import get_logger
from miai_pipeline.context import PipelineContext
from miai_pipeline.exceptions import StageError
from miai_pipeline.stage import PipelineStage
from miai_registration.apply import apply_transform
from miai_registration.register import RegistrationConfig, register_images
from miai_registration.transform_io import write_transform

logger = get_logger(__name__)


class RegistrationStageConfig(MIAIBaseConfig):
    """Configuration for :class:`RegistrationStage`.

    Attributes:
        fixed_image_path: The common reference image every case is
            registered onto (e.g. an atlas/template, or one subject
            chosen as the reference).
        output_dir: Directory registered images (and, if configured,
            registered labels) are written to.
        transform_dir: Directory estimated transforms are written to
            (one ``.tfm`` file per case).
        registration: Registration parameters.
        context_key: Which context key holds the list of moving image
            paths to register -- typically ``"preprocessed_paths"``.
        label_context_key: Optional context key holding a list of label
            paths, aligned index-for-index with ``context_key``. If
            set, each case's label is resampled with that case's
            estimated transform (nearest-neighbor interpolation), so it
            lands in the fixed image's space alongside the registered
            image.
    """

    fixed_image_path: str
    output_dir: str
    transform_dir: str
    registration: RegistrationConfig = RegistrationConfig()
    context_key: str = "preprocessed_paths"
    label_context_key: str | None = None


class RegistrationStage(PipelineStage):
    """Register every case onto a common fixed reference image.

    Typically runs after
    :class:`~miai_pipeline.stages.preprocessing.PreprocessingStage` and
    before :class:`~miai_pipeline.stages.dataset.DatasetStage`, e.g. to
    align every case to a shared atlas/template before building the
    train/val/test manifest -- a common pattern in population studies
    and multi-site data.

    Reads:
        ``<config.context_key>`` (``list[Path]``, default
        ``"preprocessed_paths"``): the moving images to register.
        ``<config.label_context_key>`` (``list[Path]``, optional): a
        label mask per case, resampled with the same transform.

    Writes:
        ``registered_paths`` (``list[Path]``): each case resampled onto
        the fixed image's grid.
        ``transform_paths`` (``list[Path]``): the estimated transform
        for each case.
        ``registered_label_paths`` (``list[Path]``, only written if
        ``config.label_context_key`` is set): each case's label,
        resampled with that case's transform.
    """

    name = "registration"
    config_cls = RegistrationStageConfig

    def __init__(self, config: RegistrationStageConfig) -> None:
        self.config = config

    def run(self, context: PipelineContext) -> PipelineContext:
        moving_paths: list[Path] = context.require(self.config.context_key)
        if not moving_paths:
            raise StageError(f"'{self.config.context_key}' is empty; nothing to register.")

        labels: list[Path] | None = None
        if self.config.label_context_key is not None:
            labels = context.require(self.config.label_context_key)
            if len(labels) != len(moving_paths):
                raise StageError(
                    f"'{self.config.label_context_key}' has {len(labels)} entries but "
                    f"'{self.config.context_key}' has {len(moving_paths)}; they must be "
                    "aligned one label per case."
                )

        fixed = sitk.ReadImage(self.config.fixed_image_path)
        output_dir = ensure_dir(self.config.output_dir)
        transform_dir = ensure_dir(self.config.transform_dir)

        registered_paths: list[Path] = []
        transform_paths: list[Path] = []
        registered_label_paths: list[Path] = []

        for i, path in enumerate(moving_paths):
            logger.info("Registering %s", path)
            moving = sitk.ReadImage(str(path))
            transform, registered = register_images(fixed, moving, self.config.registration)

            stem = Path(path).name.removesuffix(".nii.gz").removesuffix(".nii")
            registered_path = output_dir / f"{stem}_registered.nii.gz"
            sitk.WriteImage(registered, str(registered_path))
            registered_paths.append(registered_path)

            transform_path = write_transform(transform, transform_dir / f"{stem}.tfm")
            transform_paths.append(transform_path)

            if labels is not None:
                label_image = sitk.ReadImage(str(labels[i]))
                registered_label = apply_transform(
                    label_image, fixed, transform, interpolator="nearest"
                )
                label_path = output_dir / f"{stem}_label_registered.nii.gz"
                sitk.WriteImage(registered_label, str(label_path))
                registered_label_paths.append(label_path)

        context.set("registered_paths", registered_paths)
        context.set("transform_paths", transform_paths)
        if labels is not None:
            context.set("registered_label_paths", registered_label_paths)
        return context
