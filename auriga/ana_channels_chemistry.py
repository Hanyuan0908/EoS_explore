"""Chemistry of the three Eos populations against the GS/E and cold-disc backdrop.

Rows 1-3: [X/Fe]-[Fe/H] density for A (merger-born disc), B (merger-born radial,
the Eos analogue) and C (pre-merger disc heated to halo orbits, the Splash
analogue).  Row 4 overlays all three as contours together with the two reference
populations, the clean GS/E debris and the z=0 cold in-situ disc.

No z=0 radial cut: the populations differ strongly in r_z0, so cutting on it
biases the comparison (see ana_channel_radial_gradient).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import config_au18 as C
import channels_au18 as ch_mod

os.makedirs(C.FIG_DIR, exist_ok=True)
ELS = ch_mod.ELS
FEH_RANGE = (-2.5, 0.8)

d = ch_mod.load()
sp = np.load(C.OUT_DIR + '/premerger_splash.npz')
ref = np.load(C.OUT_DIR + '/z0_reference_pops.npz')
good = np.isfinite(sp['eps_birth']) & np.isfinite(sp['eps_z0']) & np.isfinite(sp['z_birth'])
Csel = good & (sp['eps_birth'] > .7) & (sp['z_birth'] < ch_mod.Z_A_MAX) & (sp['eps_z0'] < .3)

# Populations shown as density panels, then as contours.
PANELS = [
    (r'A: heated disc, $|z_b|<1$', '#2166ac',
     d['feh'][d['A']], {e: d['ratios'][e][d['A']] for e in ELS}),
    (r'B: born radial, $|z_b|>3$', '#7b3294',
     d['feh'][d['B']], {e: d['ratios'][e][d['B']] for e in ELS}),
    (r'C: pre-merger Splash', '#e08214',
     sp['feh'][Csel], {e: sp[e.lower() + 'fe'][Csel] for e in ELS}),
]
REFS = [
    ('GS/E debris', '#1a9850', ref['gse_feh'], {e: ref[f'gse_{e.lower()}fe'] for e in ELS}),
    ('cold in-situ disc', '0.35', ref['disc_feh'], {e: ref[f'disc_{e.lower()}fe'] for e in ELS}),
]
ALL = PANELS + REFS
for name, _, feh, _ in ALL:
    print(f'{name:28s} N={len(feh):9,d}  median [Fe/H]={np.nanmedian(feh):+.3f}')


def yrange(el):
    """Common y-range per element so all four rows and every population are comparable."""
    lo, hi = np.inf, -np.inf
    for _, _, feh, rat in ALL:
        y = rat[el]; q = np.isfinite(y) & np.isfinite(feh)
        a, b = np.percentile(y[q], [0.5, 99.5])
        lo, hi = min(lo, a), max(hi, b)
    pad = .10 * (hi - lo)
    return lo - pad, hi + pad


nrow = len(PANELS) + 1
fig, axes = plt.subplots(nrow, 6, figsize=(19.5, 3.05 * nrow + 1.4), sharex=True,
                         layout='constrained')
for j, el in enumerate(ELS):
    yr = yrange(el)
    for i, (label, color, feh, rat) in enumerate(PANELS):
        y = rat[el]; q = np.isfinite(feh) & np.isfinite(y)
        axes[i, j].hexbin(feh[q], y[q], gridsize=40, extent=(*FEH_RANGE, *yr),
                          bins='log', mincnt=1, cmap='magma')
        axes[i, j].set(xlim=FEH_RANGE, ylim=yr)
        if i == 0: axes[i, j].set_title(f'[{el}/Fe]')
        if j == 0:
            axes[i, j].set_ylabel(f'{label}\n(N={len(feh):,})\n[X/Fe]', color=color, fontsize=8.5)

    ax = axes[nrow - 1, j]
    for label, color, feh, rat in ALL:
        y = rat[el]; q = np.isfinite(feh) & np.isfinite(y)
        H, xe, ye = np.histogram2d(feh[q], y[q], bins=[80, 64],
                                   range=[list(FEH_RANGE), list(yr)])
        H = gaussian_filter(H, 1.3)
        xc = .5 * (xe[:-1] + xe[1:]); yc = .5 * (ye[:-1] + ye[1:])
        ls = '--' if label in ('GS/E debris', 'cold in-situ disc') else '-'
        ax.contour(xc, yc, H.T, levels=np.asarray([.15, .4, .7]) * H.max(),
                   colors=color, linewidths=1.4, linestyles=ls)
    ax.set(xlim=FEH_RANGE, ylim=yr, xlabel='[Fe/H]')
    if j == 0: ax.set_ylabel('all populations\n[X/Fe]', fontsize=8.5)
    if j == 5:
        for label, color, _, _ in ALL:
            ls = '--' if label in ('GS/E debris', 'cold in-situ disc') else '-'
            ax.plot([], [], color=color, ls=ls, label=label)
        ax.legend(loc='best', fontsize=7.2)

fig.suptitle('Au18 chemistry: the three Eos populations against the GS/E debris and the cold disc',
             fontsize=13)
out = C.FIG_DIR + '/au18_eos_channels_chemistry_clean.png'
fig.savefig(out, dpi=150)
print('saved', out)
