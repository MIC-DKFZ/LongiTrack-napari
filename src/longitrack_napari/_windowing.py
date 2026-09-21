"""CT window/level presets, in Hounsfield units.

Radiology convention: a window is a width and a centre, not a pair of bounds.
napari wants the bounds, so everything here converts between the two.
"""

from __future__ import annotations

from typing import NamedTuple


class WindowLevel(NamedTuple):
    window: int  # width, HU
    level: int  # centre, HU

    @property
    def limits(self) -> tuple[float, float]:
        half = self.window / 2
        return float(self.level - half), float(self.level + half)


def from_limits(low: float, high: float) -> WindowLevel:
    return WindowLevel(round(high - low), round((high + low) / 2))


# Narrow windows raise lesion conspicuity; the wider ones are the usual reading windows.
PRESETS: dict[str, WindowLevel] = {
    "Lesion": WindowLevel(200, 50),
    "Abdomen": WindowLevel(400, 50),
    "Liver": WindowLevel(150, 60),
    "Lung": WindowLevel(1500, -600),
    "Bone": WindowLevel(2000, 400),
    "Brain": WindowLevel(80, 40),
}

# what a scan opens with
DEFAULT_PRESET = "Lesion"
DEFAULT_WINDOW = PRESETS[DEFAULT_PRESET]

# shown when the numbers match no preset, and for non-CT data where HU means nothing
CUSTOM = "Custom"
FULL_RANGE = "Full range"


def preset_for(window_level: WindowLevel) -> str:
    for name, preset in PRESETS.items():
        if preset == window_level:
            return name
    return CUSTOM
