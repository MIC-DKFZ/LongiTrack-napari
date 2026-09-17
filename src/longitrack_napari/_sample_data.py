from __future__ import annotations

from .pantrack import DEFAULT_PATIENT, ScanPair, download_pair

CT_WINDOW = (-150.0, 250.0)


def pair_to_layers(pair: ScanPair) -> list[tuple]:
    from ._reader import read_volume

    if pair.baseline_image is None or pair.followup_image is None:
        raise ValueError(f"{pair.label} has not been downloaded yet; call download_pair first.")

    layers = []
    for path, role, identifier in (
        (pair.baseline_image, "baseline", pair.baseline),
        (pair.followup_image, "follow-up", pair.followup),
    ):
        array, kwargs = read_volume(path)
        kwargs["name"] = f"{role} - {identifier}"
        kwargs["colormap"] = "gray"
        kwargs["contrast_limits"] = CT_WINDOW
        layers.append((array, kwargs, "image"))
    return layers


def make_pantrack_pair() -> list[tuple]:
    # napari calls this on the GUI thread, so it blocks while the scans download. The
    # widget's own PanTrack button does the same thing on its worker thread instead.
    return pair_to_layers(download_pair(patient=DEFAULT_PATIENT, progress=print))
