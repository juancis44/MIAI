"""Expands case-level data dicts into one dict per slice.

:mod:`miai_segmentation.two_d`/`.two_half_d` train and run inference at
the slice level, not the whole-volume level
:func:`~miai_datasets.manifest.manifest_split_to_data_dicts` produces.
:func:`expand_to_slice_dicts` bridges the two: it reads each case's
depth (number of slices) directly from file headers -- via
:class:`SimpleITK.ImageFileReader`'s ``ReadImageInformation``, which
reads only metadata, not pixel data -- and emits one dict per
``(case, slice_index)`` pair, ready for
:func:`miai_datasets.loaders.build_dataset` once
:class:`~miai_transforms.slice_transforms.ExtractSliced` or
:class:`~miai_transforms.slice_transforms.ExtractSliceStackd` is in the
transform pipeline to act on ``"slice_index"``.
"""

from __future__ import annotations

import SimpleITK as sitk

from miai_datasets.exceptions import DatasetBuildError


def _read_depth(path: str) -> int:
    """Read a volume's depth (number of slices) from its file header only."""
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.ReadImageInformation()
    size = reader.GetSize()
    if len(size) < 3:
        raise DatasetBuildError(
            f"Expected a 3D volume (for slice extraction) at {path!r}, got size {size!r}."
        )
    # SimpleITK's GetSize() is (width, height, depth) -- the reverse of
    # GetArrayFromImage's (depth, height, width) numpy axis order (see
    # miai_transforms.sitk_transforms.LoadImageSitkd) -- so depth is
    # index 2, not 0. int(...): GetSize() is untyped from mypy's point
    # of view (no SimpleITK stubs), so its element type is Any.
    return int(size[2])


def expand_to_slice_dicts(
    data_dicts: list[dict[str, str]], *, image_key: str = "image"
) -> tuple[list[dict[str, str]], list[int]]:
    """Expand case-level data dicts into one dict per slice.

    Args:
        data_dicts: Case-level entries, as produced by
            :func:`~miai_datasets.manifest.manifest_split_to_data_dicts`.
            Each must have at least an ``image_key`` entry pointing to a
            3D volume file.
        image_key: Which key's file determines each case's depth
            (slice count). Every other key in a case's dict (e.g.
            ``"label"``) is assumed to be a co-registered volume of the
            same depth.

    Returns:
        A tuple ``(slice_dicts, case_slice_counts)``:

        - ``slice_dicts``: one dict per slice, in case-major,
          ascending-``slice_index`` order (case 0's slices, then case
          1's, ...) -- every entry from the source case dict is copied
          through unchanged, plus a new ``"slice_index"`` (stringified,
          matching this module's ``dict[str, str]`` convention; consume
          it with ``int(...)``, as
          :class:`~miai_transforms.slice_transforms.ExtractSliced`
          does).
        - ``case_slice_counts``: the number of slices contributed by
          each case, in the same order as ``data_dicts`` -- lets a
          caller (e.g.
          :class:`~miai_pipeline.stages.inference.InferenceStage`)
          regroup a flat, slice-ordered prediction stream back into
          one output per case.

    Raises:
        DatasetBuildError: If ``data_dicts`` is empty, an entry is
            missing ``image_key``, or ``image_key``'s file is not (at
            least) 3D.
    """
    if not data_dicts:
        raise DatasetBuildError("Cannot expand an empty list of data dicts into slices.")

    slice_dicts: list[dict[str, str]] = []
    case_slice_counts: list[int] = []
    for case in data_dicts:
        if image_key not in case:
            raise DatasetBuildError(f"Data dict is missing required key {image_key!r}: {case!r}")
        depth = _read_depth(case[image_key])
        if depth < 1:
            raise DatasetBuildError(f"Volume at {case[image_key]!r} has zero slices.")
        case_slice_counts.append(depth)
        for z in range(depth):
            slice_dicts.append({**case, "slice_index": str(z)})

    return slice_dicts, case_slice_counts
