"""Figure 4: three populations in [Fe/H]-[Mg/Fe]-[Al/Fe] space.
Left: column-normalised [Mg/Fe]-[Fe/H]. Middle: same, coloured by mean [Al/Fe].
Right: slice -0.8<[Fe/H]<-0.6 in [Al/Fe]-[Mg/Fe] (high-a / GS-E / Eos clumps).
"""
import _bootstrap  # noqa
import numpy as np
import matplotlib.pyplot as plt

from eos import loaders, selections as S, plotting as P, config as C

t = loaders.load_apogee()
feh = np.asarray(t["feh"]); mgfe = np.asarray(t["mgfe"]); alfe = np.asarray(t["alfe"])

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
rng = [[-1.6, 0.1], [-0.1, 0.45]]

P.column_normalised(ax[0], feh, mgfe, bins=120, range_=rng)
ax[0].set_xlabel("[Fe/H]"); ax[0].set_ylabel("[Mg/Fe]")
ax[0].set_title("Column-normalised density")

m = P.mean_in_bins(ax[1], feh, mgfe, alfe, bins=120, range_=rng, vmin=-0.2, vmax=0.3)
fig.colorbar(m, ax=ax[1], label="mean [Al/Fe]")
ax[1].set_xlabel("[Fe/H]"); ax[1].set_ylabel("[Mg/Fe]"); ax[1].set_title("Mean [Al/Fe]")

# Right: constant-metallicity slice
sl = (feh > -0.8) & (feh < -0.6)
ax[2].hexbin(alfe[sl], mgfe[sl], gridsize=45, extent=(-0.6, 0.45, -0.1, 0.4),
             cmap="viridis", bins="log")
ax[2].set_xlabel("[Al/Fe]"); ax[2].set_ylabel("[Mg/Fe]")
ax[2].set_title(r"$-0.8<$[Fe/H]$<-0.6$")
P.savefig(fig, loaders.repo_path(C.FIGURE_DIR, "fig04_3pop_space.png"))
