"""Figure 2: column-normalised [Al/Fe]-[Fe/H] for accreted / high-a / low-a."""
import _bootstrap  # noqa
import numpy as np
import matplotlib.pyplot as plt

from eos import loaders, selections as S, plotting as P, config as C

t = loaders.load_apogee()
acc = S.is_accreted(t) & (np.abs(np.asarray(t["Lz"], float)) < C.ACCRETED_LZ_MAX)
panels = [("accreted (|Lz|<500)", acc),
          (r"high-$\alpha$", S.is_highalpha(t)),
          (r"low-$\alpha$", S.is_lowalpha(t))]

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), sharex=True, sharey=True)
rng = [[-2.0, 0.4], [-0.6, 0.6]]
for a, (name, m) in zip(ax, panels):
    P.column_normalised(a, np.asarray(t["feh"])[m], np.asarray(t["alfe"])[m],
                        bins=120, range_=rng)
    a.axhline(C.AL_INSITU, color="w", ls="--", lw=1)
    a.set_xlabel("[Fe/H]"); a.set_title(name)
ax[0].set_ylabel("[Al/Fe]")
P.savefig(fig, loaders.repo_path(C.FIGURE_DIR, "fig02_alfe_pops.png"))
