"""Phase E3: orbital dynamics of Eos vs the high-alpha Splash (APOGEE; large N).

If the same GS/E event heated both, Eos and Splash should share r_apo / ecc /
Lz distributions (heated scenario). Distinct or disc-connected structure favours
the onset scenario.  Uses APOGEE (precise orbits, many stars).
"""
import _bootstrap  # noqa
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from eos import loaders, selections as S, plotting as P, config as C

t = loaders.load_apogee()
eos = t[S.eos_mask(t)]
spl = t[S.splash_mask(t)]
disc = t[S.lowalpha_disc_mask(t)]

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
specs = [("rap", "$r_{apo}$ [kpc]", np.linspace(0, 20, 30)),
         ("ecc", "eccentricity", np.linspace(0, 1, 30)),
         ("Lz3", "$L_z\\times10^{-3}$", np.linspace(-1, 2.5, 35))]
for a, (col, lab, bins) in zip(ax, specs):
    for samp, c, l in [(disc, "tab:gray", "low-$\\alpha$ disc"),
                       (eos, "k", "Eos"), (spl, "tab:blue", "Splash")]:
        a.hist(np.asarray(samp[col], float), bins=bins, density=True,
               histtype="step", color=c, lw=2, label=l)
    a.set_xlabel(lab); a.set_ylabel("normalised")
ax[0].legend(fontsize=8)
P.savefig(fig, loaders.repo_path(C.FIGURE_DIR, "ana3_orbits_vs_splash.png"))

for col in ["rap", "ecc", "Lz3"]:
    e = np.asarray(eos[col], float); s = np.asarray(spl[col], float)
    e = e[np.isfinite(e)]; s = s[np.isfinite(s)]
    ks = ks_2samp(e, s)
    print(f"{col:5s}  Eos med={np.median(e):6.2f}  Splash med={np.median(s):6.2f}  "
          f"KS D={ks.statistic:.3f} p={ks.pvalue:.2e}")
