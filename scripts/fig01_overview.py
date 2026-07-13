"""Figure 1: overview of the base APOGEE sample.
Left: E-Lz density. Middle: [Mg/Fe]-[Fe/H] with selection lines.
Right: [Al/Fe]-[Fe/H] with the in-situ/accreted line.
"""
import _bootstrap  # noqa
import numpy as np
import matplotlib.pyplot as plt

from eos import loaders, selections as S, plotting as P, config as C

t = loaders.load_apogee()
feh = np.asarray(t["feh"]); mgfe = np.asarray(t["mgfe"]); alfe = np.asarray(t["alfe"])

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

# --- Left: E-Lz ---
P.log_density(ax[0], t["Lz3"], t["E5"], bins=300, range_=[[-2.2, 2.2], [-0.95, 0.05]])
ax[0].axvline(0, color="k", lw=1)
ax[0].set_xlabel(r"$L_z\,\times10^{-3}$"); ax[0].set_ylabel(r"$E\,\times10^{-5}$")
ax[0].set_title("Energy, Lz")

# --- Middle: [Mg/Fe]-[Fe/H] with selection lines ---
P.log_density(ax[1], feh, mgfe, bins=300, range_=[[-2.0, 0.5], [-0.1, 0.5]])
xx = np.linspace(-2.0, 0.5, 100)
ax[1].plot(xx, S.mgfe_split(xx), "w--", lw=1.5, label="high/low-$\\alpha$ split")
ax[1].set_xlabel("[Fe/H]"); ax[1].set_ylabel("[Mg/Fe]")
ax[1].set_title("Magnesium"); ax[1].legend(loc="upper right", fontsize=8)

# --- Right: [Al/Fe]-[Fe/H] with in-situ/accreted line ---
P.log_density(ax[2], feh, alfe, bins=300, range_=[[-2.0, 0.5], [-0.6, 0.6]])
ax[2].axhline(C.AL_INSITU, color="w", ls="--", lw=1.5, label="in-situ / accreted")
ax[2].set_xlabel("[Fe/H]"); ax[2].set_ylabel("[Al/Fe]")
ax[2].set_title("Aluminium"); ax[2].legend(loc="lower right", fontsize=8)

P.savefig(fig, loaders.repo_path(C.FIGURE_DIR, "fig01_overview.png"))
