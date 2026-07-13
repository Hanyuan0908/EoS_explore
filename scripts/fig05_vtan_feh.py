"""Figure 5: Vtan vs [Fe/H] for the low-alpha sample, coloured by r_apo / r_peri / age.
Shows the thin disk (high Vtan), bulge/bar ([Fe/H]>0, low Vtan), and Eos (Vtan~0,
metal-poor, large r_apo).
"""
import _bootstrap  # noqa
import numpy as np
import matplotlib.pyplot as plt

from eos import loaders, selections as S, plotting as P, config as C

t = loaders.load_apogee()
low = t[S.is_lowalpha(t)]
feh = np.asarray(low["feh"]); vtan = np.asarray(low["vtan"])

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), sharex=True, sharey=True)
specs = [("rap", "$r_{apo}$ [kpc]", 1, 17, "viridis"),
         ("rperi", "$r_{peri}$ [kpc]", 1, 12, "viridis"),
         ("age", "Age [Gyr]", 3, 12, "plasma")]
for a, (col, lab, vmin, vmax, cmap) in zip(ax, specs):
    c = np.asarray(low[col])
    sel = np.isfinite(c)
    if col == "age":  # apply relative age-error cut
        rel = np.asarray(low["age_err"]) / np.where(c > 0, c, np.nan)
        sel &= np.isfinite(rel) & (rel < C.AGE_REL_ERR_MAX)
    sc = a.scatter(feh[sel], vtan[sel], c=c[sel], s=2, vmin=vmin, vmax=vmax,
                   cmap=cmap, rasterized=True)
    fig.colorbar(sc, ax=a, label=lab)
    a.set_xlabel("[Fe/H]"); a.set_xlim(-1.5, 0.5); a.set_ylim(-200, 320)
ax[0].set_ylabel(r"$V_{tan}$ [km/s]")
P.savefig(fig, loaders.repo_path(C.FIGURE_DIR, "fig05_vtan_feh.png"))
