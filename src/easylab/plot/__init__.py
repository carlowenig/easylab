import matplotlib
from .figure import *
from .plot import *

matplotlib.rcParams.update(
    {
        # "interactive": True,
        "figure.dpi": 200,
        "figure.figsize": (7, 4),
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.grid": True,
        "axes.labelsize": 12,
        "grid.linewidth": 0.2,
        "grid.alpha": 0.5,
        "lines.linewidth": 0.7,
        "lines.dashed_pattern": (6, 4),
    }
)
