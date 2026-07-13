"""Phase E1: LAMOST MSTO age distributions of Eos vs low-alpha disc vs Splash.

Heated-disc scenario  -> Eos ~ Splash, sharp old cutoff near the GS/E merger epoch.
Onset scenario        -> Eos older, the leading (oldest) edge of the low-alpha AMR.
"""
import _bootstrap  # noqa
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

from eos import loaders, selections as S, plotting as P, config as C

m = Table.read(loaders.repo_path(C.RESULTS_DIR, "lamost_eos_sample.fits"))
age = np.asarray(m["age"], float)
rel = np.asarray(m["age_err"], float) / np.where(age > 0, age, np.nan)
good = np.isfinite(age) & (rel < C.AGE_REL_ERR_MAX)

pops = [("low-$\\alpha$ disc", S.lowalpha_disc_mask(m) & good, "tab:red", "--"),
        ("Eos", S.eos_mask(m) & good, "tab:red", "-"),
        ("Splash", S.splash_mask(m) & good, "tab:blue", "-")]

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
bins = np.linspace(0, 14, 29)
verdict = []
for name, sel, col, ls in pops:
    a = age[sel]
    ax[0].hist(a, bins=bins, density=True, histtype="step", color=col, ls=ls, lw=2,
               label=f"{name} (n={sel.sum()}, med={np.median(a):.1f})")
    verdict.append((name, np.median(a), np.percentile(a, 16), np.percentile(a, 84)))
ax[0].set_xlabel("Age [Gyr]"); ax[0].set_ylabel("normalised")
ax[0].set_title("LAMOST MSTO age distributions"); ax[0].legend(fontsize=8)

# cumulative for sharper comparison of Eos vs Splash
for name, sel, col, ls in pops:
    a = np.sort(age[sel])
    ax[1].plot(a, np.linspace(0, 1, len(a)), color=col, ls=ls, lw=2, label=name)
ax[1].set_xlabel("Age [Gyr]"); ax[1].set_ylabel("cumulative"); ax[1].set_title("CDF")
ax[1].legend(fontsize=8)
P.savefig(fig, loaders.repo_path(C.FIGURE_DIR, "ana1_age_distributions.png"))

# KS tests
from scipy.stats import ks_2samp
eos_a = age[S.eos_mask(m) & good]
spl_a = age[S.splash_mask(m) & good]
dsk_a = age[S.lowalpha_disc_mask(m) & good]
print("median ages:", {n: round(med, 2) for n, med, _, _ in verdict})
print("KS Eos vs Splash:", ks_2samp(eos_a, spl_a))
print("KS Eos vs low-a disc:", ks_2samp(eos_a, dsk_a))
