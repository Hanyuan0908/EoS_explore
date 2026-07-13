"""Shared matplotlib helpers for consistent figure style."""
from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.labelsize": 12,
    "image.cmap": "viridis",
})


def log_density(ax, x, y, bins, range_, cmap="viridis"):
    """Logarithmic 2D density (like the paper's E-Lz / abundance panels)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    H, xe, ye = np.histogram2d(x[m], y[m], bins=bins, range=range_)
    H = np.log10(H.T + 1.0)
    return ax.pcolormesh(xe, ye, H, cmap=cmap, shading="auto")


def column_normalised(ax, x, y, bins, range_, cmap="viridis"):
    """Column-normalised density: each x-column scaled to its own max (paper Fig 2)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    H, xe, ye = np.histogram2d(x[m], y[m], bins=bins, range=range_)
    colmax = H.max(axis=1, keepdims=True)
    colmax[colmax == 0] = 1.0
    Hn = (H / colmax).T
    return ax.pcolormesh(xe, ye, Hn, cmap=cmap, shading="auto")


def mean_in_bins(ax, x, y, c, bins, range_, cmap="viridis", vmin=None, vmax=None):
    """2D map coloured by the mean of a third quantity c (paper Fig 4 middle)."""
    x = np.asarray(x, float); y = np.asarray(y, float); c = np.asarray(c, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    x, y, c = x[m], y[m], c[m]
    sums, xe, ye = np.histogram2d(x, y, bins=bins, range=range_, weights=c)
    cnt, _, _ = np.histogram2d(x, y, bins=bins, range=range_)
    with np.errstate(invalid="ignore"):
        M = np.where(cnt > 0, sums / cnt, np.nan).T
    return ax.pcolormesh(xe, ye, M, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")
