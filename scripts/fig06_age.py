"""Figure 6: age-metallicity behaviour and age distributions (APOGEE astroNN ages).
Left: median age vs [Fe/H] for high-a (blue) and low-a (red).
Middle: age vs [Fe/H] for Vtan<80, coloured by r_apo.
Right: age distributions of low-a disc, Eos, Splash (the key Splash~Eos comparison).
NB: APOGEE ages are unreliable >10 Gyr / below [Fe/H]=-1 (paper caveat); LAMOST
MSTO ages (Phase D/E) are the primary age diagnostic.
"""
import _bootstrap  # noqa
import numpy as np
import matplotlib.pyplot as plt

from eos import loaders, selections as S, plotting as P, config as C


def good_age(tab):
    a = np.asarray(tab["age"]); e = np.asarray(tab["age_err"])
    rel = e / np.where(a > 0, a, np.nan)
    return np.isfinite(a) & np.isfinite(rel) & (rel < C.AGE_REL_ERR_MAX)


def median_track(feh, age, edges):
    idx = np.digitize(feh, edges)
    cen, med = [], []
    for i in range(1, len(edges)):
        m = idx == i
        if m.sum() > 20:
            cen.append(0.5 * (edges[i - 1] + edges[i]))
            med.append(np.median(age[m]))
    return np.array(cen), np.array(med)


t = loaders.load_apogee()
ga = good_age(t)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
edges = np.linspace(-1.5, 0.5, 26)

# Left: AMR tracks
for mask, col, lab in [(S.is_highalpha(t), "tab:blue", r"high-$\alpha$"),
                       (S.is_lowalpha(t), "tab:red", r"low-$\alpha$")]:
    sel = mask & ga
    c, mtrk = median_track(np.asarray(t["feh"])[sel], np.asarray(t["age"])[sel], edges)
    ax[0].plot(c, mtrk, "-", color=col, lw=2, label=lab)
ax[0].set_xlabel("[Fe/H]"); ax[0].set_ylabel("Age [Gyr]")
ax[0].set_title("Age-metallicity"); ax[0].legend()

# Middle: low-Vtan age-[Fe/H] coloured by r_apo
lv = (np.abs(np.asarray(t["vtan"])) < 80) & ga & S.is_lowalpha(t)
sc = ax[1].scatter(np.asarray(t["feh"])[lv], np.asarray(t["age"])[lv],
                   c=np.asarray(t["rap"])[lv], s=4, vmin=1, vmax=17, cmap="viridis")
fig.colorbar(sc, ax=ax[1], label="$r_{apo}$ [kpc]")
ax[1].set_xlabel("[Fe/H]"); ax[1].set_ylabel("Age [Gyr]")
ax[1].set_title(r"low-$\alpha$, $V_{tan}<80$")

# Right: age distributions
bins = np.linspace(0, 13, 27)
for mask, col, lab, ls in [(S.lowalpha_disc_mask(t), "tab:red", r"low-$\alpha$ disc", "--"),
                           (S.eos_mask(t), "tab:red", "Eos", "-"),
                           (S.splash_mask(t), "tab:blue", "Splash", "-")]:
    sel = mask & ga
    ax[2].hist(np.asarray(t["age"])[sel], bins=bins, density=True, histtype="step",
               color=col, ls=ls, lw=2, label=f"{lab} (n={int(sel.sum())})")
ax[2].set_xlabel("Age [Gyr]"); ax[2].set_ylabel("normalised")
ax[2].set_title("Age distributions"); ax[2].legend(fontsize=8)
P.savefig(fig, loaders.repo_path(C.FIGURE_DIR, "fig06_age.png"))
