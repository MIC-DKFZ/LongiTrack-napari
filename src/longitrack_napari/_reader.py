from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

SUPPORTED_SUFFIXES = (".nii", ".nii.gz", ".nrrd", ".mha", ".mhd", ".gipl", ".gipl.gz")


def _is_supported(path: str) -> bool:
    lowered = str(path).lower()
    return any(lowered.endswith(suffix) for suffix in SUPPORTED_SUFFIXES)


def _strip_suffix(name: str) -> str:
    for suffix in sorted(SUPPORTED_SUFFIXES, key=len, reverse=True):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def read_volume(path: str | Path) -> tuple[np.ndarray, dict]:
    # read through SimpleITKIO rather than SimpleITK directly, so the array a point is
    # clicked on is byte for byte the array the network is preprocessed from
    from longiseg.imageio.simpleitk_reader_writer import SimpleITKIO

    path = Path(path)
    data, properties = SimpleITKIO().read_images([str(path)])
    array = data[0]

    spacing_zyx = tuple(float(abs(s)) for s in reversed(properties["sitk_stuff"]["spacing"]))
    while len(spacing_zyx) < array.ndim:
        spacing_zyx = (1.0, *spacing_zyx)

    kwargs = {
        "name": _strip_suffix(path.name),
        "scale": spacing_zyx[-array.ndim :],
        "metadata": {
            # absolute but deliberately not resolved: in the Hugging Face cache the file
            # is a .nii.gz symlink onto an extension-less blob, and both ITK and SimpleITK
            # pick their reader from the extension
            "longitrack_path": os.path.abspath(path),
            "spacing_zyx": spacing_zyx,
            "sitk_stuff": properties["sitk_stuff"],
        },
    }
    return array, kwargs


def napari_get_reader(path: str | Sequence[str]):
    paths = [path] if isinstance(path, str) else list(path)
    if not paths or not all(_is_supported(p) for p in paths):
        return None
    return _reader


def _reader(path: str | Sequence[str]) -> list[tuple]:
    paths = [path] if isinstance(path, str) else list(path)
    layers = []
    for single in paths:
        array, kwargs = read_volume(single)
        is_label = (
            np.issubdtype(array.dtype, np.integer)
            and array.ndim == 3
            and int(array.max()) <= 255
            and len(np.unique(array)) <= 16
        )
        layers.append((array, kwargs, "labels" if is_label else "image"))
    return layers
