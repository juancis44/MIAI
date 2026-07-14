"""SimpleITK-backed MONAI transforms.

MIAI standardizes on SimpleITK (already a core dependency, used
throughout :mod:`miai_dicom` and :mod:`miai_pipeline`) and PyTorch for
all image I/O and tensor computation, rather than pulling in an
additional reader backend (``nibabel`` or ``itk``) just to satisfy
MONAI's own ``LoadImage``. This module provides
:class:`LoadImageSitkd`, a MONAI dictionary transform that reads NIfTI
(or any SimpleITK-readable) files directly with SimpleITK.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np
import SimpleITK as sitk
from monai.config import KeysCollection
from monai.transforms import MapTransform


class LoadImageSitkd(MapTransform):
    """Loads image files into channel-first arrays using SimpleITK.

    For each of ``keys``, reads the file path stored at that key with
    :func:`SimpleITK.ReadImage`, converts it to a ``float32`` numpy
    array via :func:`SimpleITK.GetArrayFromImage` (SimpleITK's own
    ``(D, H, W)`` axis convention), and adds a leading channel
    dimension, producing a ``(1, D, H, W)`` array -- no separate
    "ensure channel first" step needed. The image's spacing, origin,
    and direction are stashed under ``f"{key}_meta_dict"`` for any
    downstream code that needs them; MIAI's reference transform
    pipeline does not, since spacing/orientation are handled upstream
    by :class:`~miai_pipeline.stages.preprocessing.PreprocessingStage`.

    Keeping every array in SimpleITK's axis convention end to end means
    predictions can be written back out with
    :func:`SimpleITK.GetImageFromArray` plus
    ``image.CopyInformation(reference_image)``, with no axis
    transposition needed -- see
    :func:`miai_segmentation.infer.run_inference`.
    """

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            path = str(d[key])
            image = sitk.ReadImage(path)
            array = sitk.GetArrayFromImage(image).astype(np.float32)
            d[key] = array[np.newaxis, ...]
            d[f"{key}_meta_dict"] = {
                "filename_or_obj": path,
                "spacing": image.GetSpacing(),
                "origin": image.GetOrigin(),
                "direction": image.GetDirection(),
            }
        return d
