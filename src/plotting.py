"""Shared matplotlib style: fixed categorical order, thin marks, recessive grid."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference categorical palette, light mode, assigned in fixed order (never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SEQ = "Blues"

# Stable colour per entity (colour follows the method, not its rank).
METHOD_COLORS = {}


def color_for(name: str) -> str:
    if name not in METHOD_COLORS:
        METHOD_COLORS[name] = SERIES[len(METHOD_COLORS) % len(SERIES)]
    return METHOD_COLORS[name]


SPLIT_COLORS = {"IN_DISTRIBUTION": SERIES[0], "NEAR_OOD": SERIES[1], "NEAR_OOD_ADV": SERIES[3],
                "FAR_OOD": SERIES[2]}
SPLIT_NAMES = {"IN_DISTRIBUTION": "ID", "NEAR_OOD": "Near-OOD", "NEAR_OOD_ADV": "Near-OOD adv.",
               "FAR_OOD": "Far-OOD"}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "text.color": TEXT,
        "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


apply_style()
