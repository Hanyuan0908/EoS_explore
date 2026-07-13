"""Figure 3: E-Lz of the three chemically-selected populations.
Key validation: the Eos clump must appear at |Lz|~0, E*1e-5 ~ -0.55 in the
low-alpha panel.
"""
import _bootstrap  # noqa
import numpy as np
import matplotlib.pyplot as plt

from eos import loaders, selections as S, plotting as P, config as C

t = loaders.load_apogee()
masks = [("accreted", S.is_accreted(t)),
         (r"high-$\alpha$", S.is_highalpha(t)),
         (r"low-$\alpha$", S.is_lowalpha(t))]

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), sharex=True, sharey=True)
rng = [[-2.2, 2.2], [-1.5, 0.05]]
for a, (name, m) in zip(ax, masks):
    P.log_density(a, t["Lz3"][m], t["E5"][m], bins=250, range_=rng)
    a.axvline(0, color="k", lw=0.8)
    a.set_xlabel(r"$L_z\,\times10^{-3}$"); a.set_title(name)
ax[0].set_ylabel(r"$E\,\times10^{-5}$")

# mark the Eos locus on the low-alpha panel
e = t[S.eos_mask(t)]
ax[2].scatter(e["Lz3"], e["E5"], s=4, c="red", alpha=0.5, label=f"Eos (n={len(e)})")
ax[2].legend(loc="upper left", fontsize=9)

P.savefig(fig, loaders.repo_path(C.FIGURE_DIR, "fig03_elz_pops.png"))
print("Eos median E5=%.3f Lz3=%.3f" % (np.nanmedian(e["E5"]), np.nanmedian(e["Lz3"])))
