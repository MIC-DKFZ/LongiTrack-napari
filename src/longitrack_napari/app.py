from __future__ import annotations

import argparse
from pathlib import Path


def launch(baseline: str | Path | None = None, followup: str | Path | None = None, block: bool = False):
    """Open napari with the tracking widget docked, optionally on a pair of scans.

        from longitrack_napari.app import launch

        viewer, widget = launch("baseline.nii.gz", "followup.nii.gz", block=True)

    Returns the viewer and the widget. With `block=False` the caller is responsible
    for starting the Qt event loop, e.g. with `napari.run()`.
    """
    import napari

    from ._widget import LongiTrackWidget

    viewer = napari.Viewer()
    widget = LongiTrackWidget(viewer)
    viewer.window.add_dock_widget(widget, name="LongiSeg Tracking", area="right")

    for role, scan in (("baseline", baseline), ("followup", followup)):
        if scan is not None:
            widget.open_scan(role, scan)

    if block:
        napari.run()
    return viewer, widget


def main() -> None:
    """Launch napari with the LongiTrack widget already docked."""
    parser = argparse.ArgumentParser(description="Launch the LongiTrack napari viewer.")
    parser.add_argument("--baseline", type=Path, help="optional baseline scan")
    parser.add_argument("--followup", type=Path, help="optional follow-up scan")
    args = parser.parse_args()
    launch(args.baseline, args.followup, block=True)
