"""Phase E2 (DECISIVE TEST): age at fixed [Fe/H], halo-orbit Eos vs disc-orbit low-a.

Within the low-alpha sequence, compare the MSTO age of halo-orbit (Eos) vs
disc-orbit stars in matched [Fe/H] bins across the Eos regime.

  onset scenario  -> Eos systematically OLDER at fixed [Fe/H] (born first, before
                     the disc spun up).
  heated scenario -> SAME age at fixed [Fe/H] (one population, merely kicked).
"""
import _bootstrap  # noqa
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

from eos import loaders, selections as S, plotting as P, config as C

m = Table.read(loaders.repo_path(C.RESULTS_DIR, "lamost_eos_sample.fits"))
age = np.asarray(m["age"], float)
feh = np.asarray(m["feh"], float)
rel = np.asarray(m["age_err"], float) / np.where(age > 0, age, np.nan)
good = np.isfinite(age) & np.isfinite(feh) & (rel < C.AGE_REL_ERR_MAX)

low = S.is_lowalpha(m)
halo = low & S.is_halo_orbit(m) & good      # Eos-like
disc = low & S.is_disc_orbit(m) & good      # rotating low-alpha

edges = np.array([-1.1, -0.95, -0.8, -0.65, -0.5, -0.35, -0.2])
cen = 0.5 * (edges[:-1] + edges[1:])

def track(mask):
    med, lo, hi, n = [], [], [], []
    for i in range(len(cen)):
        s = mask & (feh >= edges[i]) & (feh < edges[i + 1])
        a = age[s]
        if len(a) >= 8:
            med.append(np.median(a)); lo.append(np.percentile(a, 25))
            hi.append(np.percentile(a, 75)); n.append(len(a))
        else:
            med.append(np.nan); lo.append(np.nan); hi.append(np.nan); n.append(len(a))
    return np.array(med), np.array(lo), np.array(hi), np.array(n)

hm, hlo, hhi, hn = track(halo)
dm, dlo, dhi, dn = track(disc)

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.fill_between(cen, dlo, dhi, color="tab:red", alpha=0.15)
ax.fill_between(cen, hlo, hhi, color="k", alpha=0.12)
ax.plot(cen, dm, "o-", color="tab:red", lw=2, label="disc-orbit low-$\\alpha$")
ax.plot(cen, hm, "s-", color="k", lw=2, label="halo-orbit (Eos)")
for x, y, nn in zip(cen, hm, hn):
    if np.isfinite(y): ax.annotate(f"{nn}", (x, y), textcoords="offset points",
                                   xytext=(0, 8), fontsize=8, ha="center")
ax.set_xlabel("[Fe/H]"); ax.set_ylabel("median MSTO age [Gyr]")
ax.set_title("Age at fixed [Fe/H]: Eos vs disc-orbit low-$\\alpha$")
ax.legend()
P.savefig(fig, loaders.repo_path(C.FIGURE_DIR, "ana2_amr_fixed_feh.png"))

print("bin_center  age_halo(Eos)  age_disc   delta(Eos-disc)  n_halo n_disc")
for i in range(len(cen)):
    d = hm[i] - dm[i]
    print(f"  {cen[i]:+.2f}      {hm[i]:6.2f}        {dm[i]:6.2f}     {d:+6.2f}"
          f"        {hn[i]:4d}  {dn[i]:5d}")
fin = np.isfinite(hm) & np.isfinite(dm)
print(f"\nmean age offset (Eos - disc) over bins: {np.nanmean((hm-dm)[fin]):+.2f} Gyr")
