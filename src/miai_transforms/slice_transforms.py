"""Slice-extraction MONAI transforms, for 2D/2.5D segmentation.

:mod:`miai_segmentation.two_d` and :mod:`miai_segmentation.two_half_d`
operate per-slice rather than on a whole volume. These transforms take
an already-loaded ``(C, D, H, W)`` volume array (as produced by
:class:`~miai_transforms.sitk_transforms.LoadImageSitkd`) and reduce it
to the 2D input a slice-level model expects, using a ``"slice_index"``
entry that :func:`miai_datasets.slices.expand_to_slice_dicts` adds to
each data dict before the transform pipeline runs. Kept in their own
module (rather than :mod:`~miai_transforms.sitk_transforms`) because
neither transform touches SimpleITK -- they only index an array already
in memory.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

from monai.config import KeysCollection
from monai.transforms import MapTransform

from miai_transforms.exceptions import TransformError


class ExtractSliced(MapTransform):
    """Extracts one 2D slice from a ``(C, D, H, W)`` volume array.

    For each of ``keys``, replaces the ``(C, D, H, W)`` array with its
    ``(C, H, W)`` slice at depth index ``d[index_key]`` -- MIAI's 2D
    modality's per-slice single-channel input/label shape. The index is
    clamped to the volume's valid depth range as a defensive
    fallback; in normal use it is never out of range, since
    :func:`miai_datasets.slices.expand_to_slice_dicts` only ever
    produces indices within each case's actual depth.

    Attributes:
        index_key: The data dict key holding the (per-item) slice
            index to extract.
    """

    def __init__(
        self,
        keys: KeysCollection,
        index_key: str = "slice_index",
        allow_missing_keys: bool = False,
    ) -> None:
        """Store which keys to slice and where the slice index lives."""
        super().__init__(keys, allow_missing_keys)
        self.index_key = index_key

    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        """Slice each configured key's volume array at the item's index."""
        d = dict(data)
        index = int(d[self.index_key])
        for key in self.key_iterator(d):
            volume = d[key]
            depth = volume.shape[1]
            clamped = max(0, min(index, depth - 1))
            d[key] = volume[:, clamped, :, :]
        return d


class ExtractSliceStackd(MapTransform):
    """Extracts a stack of adjacent 2D slices from a ``(C, D, H, W)`` array.

    For each of ``keys``, replaces the ``(C, D, H, W)`` array (``C`` must
    be ``1``, as produced by
    :class:`~miai_transforms.sitk_transforms.LoadImageSitkd`) with a
    ``(context_slices, H, W)`` array: ``context_slices`` adjacent depth
    indices centered on ``d[index_key]``, stacked along the channel
    axis -- MIAI's 2.5D modality's input shape (see
    :class:`~miai_segmentation.two_half_d.models.StackedUNetConfig`).
    Indices past a volume's boundary (near ``depth_index=0`` or the last
    slice) clamp to the nearest valid index (edge replication), so every
    slice in a volume -- including the first and last -- can be a center
    slice, at the cost of duplicating an edge slice's contribution in
    the stack for those cases.

    Typically applied only to the ``"image"`` key -- pair it with
    :class:`ExtractSliced` (not this transform) on the ``"label"`` key,
    since a 2.5D model still predicts a single center-slice mask, not a
    stack.

    Attributes:
        context_slices: Number of adjacent slices to stack (must be
            positive and odd, so there is a well-defined center slice).
        index_key: The data dict key holding the (per-item) center
            slice index.
    """

    def __init__(
        self,
        keys: KeysCollection,
        context_slices: int = 3,
        index_key: str = "slice_index",
        allow_missing_keys: bool = False,
    ) -> None:
        """Store the stack size, index key, and which keys to stack."""
        super().__init__(keys, allow_missing_keys)
        if context_slices < 1 or context_slices % 2 == 0:
            raise TransformError(
                f"context_slices must be a positive odd number, got {context_slices!r}."
            )
        self.context_slices = context_slices
        self.index_key = index_key
        self._half = context_slices // 2

    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        """Stack each configured key's adjacent slices, centered on the item's index."""
        d = dict(data)
        index = int(d[self.index_key])
        for key in self.key_iterator(d):
            volume = d[key]
            depth = volume.shape[1]
            indices = [
                max(0, min(index + offset, depth - 1))
                for offset in range(-self._half, self._half + 1)
            ]
            d[key] = volume[0, indices, :, :]
        return d
