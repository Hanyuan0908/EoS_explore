"""Davies+2025 Fig 14 'Eos Mg bifurcation': split the low-alpha halo region into two
branches (upper=higher Mg, lower=lower Mg) by the diagonal line
  [Mg/Fe] = 0.317*[Fe/H] + 0.353,  within the box [Fe/H] in [-0.9,-0.2], [Mg/Fe] in [0.05,0.30],
on the Davies halo sample (base & (ecc>0.7 | Lz<0)). Compare their [N/Fe] distributions and
dispersion (deconvolved sigma + P95-P5) to ask whether they are the same population.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde
REPO = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/eos-figures')
sys.path.insert(0, str(REPO))
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
from eos_figures.plotting import label_axes
rng = np.random.default_rng(0); c = Cuts()
FIG = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/figures_repro')
cat = load_catalog('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c); base = np.asarray(m['base'], bool)
feh = np.asarray(cat['fe_h'], float); mg = np.asarray(cat['mg_fe'], float); al = np.asarray(cat['al_fe'], float)
nfe = np.asarray(cat['n_fe'], float); nerr = np.asarray(cat['n_fe_err'], float)
rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float); lz = np.asarray(cat['lz'], float)
ecc = (rap - rperi) / (rap + rperi)
vphi = np.asarray(cat['galvt'], float)
halo = base & ((ecc > 0.7) | (lz < 0))
# low-alpha DISC reference: in-situ low-alpha (thin_al) on disc orbits (V_tan>150).
# Start at [Fe/H]>-0.8 -- the low-alpha disc barely exists below that (only ~35 stars in -0.9..-0.8).
disc = np.asarray(m['thin_al'], bool) & (vphi > 150) & (feh > -0.8) & (feh < -0.2) & np.isfinite(nfe)
BX = (-0.9, -0.2)
def divline(f): return 0.317 * f + 0.353           # Davies cyan divider
def acc_line(f): return c.slope_acc * f + c.inter_acc      # accreted/in-situ (below=accreted)
def hl_line(f): return c.slope_acc2 * f + c.inter_acc2     # high/low-alpha (above=high-alpha)
# LOW-ALPHA, IN-SITU wedge only: below high/low line (no Splash), above accreted line & Al>-0.12 (no GS/E)
lowa = halo & (feh > BX[0]) & (feh < BX[1]) & (mg > acc_line(feh)) & (mg < hl_line(feh)) & (al > c.alfe_cut) & np.isfinite(nfe)
upper = lowa & (mg > divline(feh)); lower = lowa & (mg <= divline(feh))
box = lowa
NRICH = 0.30
nrich = lowa & (nfe > NRICH)          # the nitrogen-enriched tail
CU, CL = '#e07a1f', '#2b6cb0'

def mad(x): x = x[np.isfinite(x)]; return 1.4826 * np.median(np.abs(x - np.median(x)))
def p95m5(x): x = x[np.isfinite(x)]; return np.percentile(x, 95) - np.percentile(x, 5)
def sigint(y, e):
    ok = np.isfinite(y) & np.isfinite(e); y, e = y[ok], e[ok]
    return np.sqrt(max(mad(y) ** 2 - np.mean(e ** 2), 0)) if y.size >= 8 else np.nan

fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
# (1) separation map
s = halo & np.isfinite(feh) & np.isfinite(mg)
h, xe, ye = np.histogram2d(feh[s], mg[s], bins=[70, 55], range=[(-1.6, 0.4), (-0.05, 0.45)])
him = np.full_like(h, np.nan); him[h > 0] = np.log10(h[h > 0])
ax[0].imshow(him.T, origin='lower', extent=[-1.6, 0.4, -0.05, 0.45], aspect='auto', cmap='Greys',
             vmin=np.nanpercentile(him, 2), vmax=np.nanpercentile(him, 99), zorder=0)
ax[0].scatter(feh[upper], mg[upper], s=7, c=CU, linewidths=0, label=f'upper (n={int(upper.sum())})', zorder=2)
ax[0].scatter(feh[lower], mg[lower], s=7, c=CL, linewidths=0, label=f'lower (n={int(lower.sum())})', zorder=2)
xx = np.linspace(*BX, 50)
ax[0].plot(xx, divline(xx), 'c--', lw=2, zorder=3, label='Davies divider')
ax[0].plot(xx, acc_line(xx), 'r--', lw=1.2, zorder=3)      # accreted line (lower bound)
ax[0].plot(xx, hl_line(xx), 'r:', lw=1.6, zorder=3)        # high/low-alpha line (upper bound)
ax[0].scatter(feh[nrich], mg[nrich], s=70, facecolors='none', edgecolors='magenta', linewidths=1.6,
              marker='o', label=f'N-rich tail ([N/Fe]>{NRICH}, n={int(nrich.sum())})', zorder=4)
ax[0].set_xlim(-1.6, 0.4); ax[0].set_ylim(-0.05, 0.45)
label_axes(ax[0], '[Fe/H]', '[Mg/Fe]', 'Eos Mg bifurcation (Davies halo)')
ax[0].legend(frameon=False, fontsize=9, loc='upper right')
# (2) N distributions
xg = np.linspace(-0.4, 0.9, 240)
for sel, col, lab in [(upper, CU, 'upper'), (lower, CL, 'lower')]:
    y = nfe[sel][np.isfinite(nfe[sel])]
    ax[1].hist(y, bins=np.linspace(-0.4, 0.9, 30), density=True, color=col, alpha=0.35)
    ax[1].plot(xg, gaussian_kde(y)(xg), color=col, lw=2,
               label=f'{lab}: med={np.median(y):+.2f}, $\\sigma$={mad(y):.3f}, P95-P5={p95m5(y):.2f}')
    for q in np.percentile(y, [5, 95]):
        ax[1].axvline(q, color=col, ls=':', lw=1)
ax[1].set_xlim(-0.4, 0.9)
label_axes(ax[1], '[N/Fe]', 'density', 'N distribution of the two branches')
ax[1].legend(frameon=False, fontsize=8.5, loc='upper right')
# (3) deconvolved sigma_N vs [Fe/H]
edges = np.arange(-0.9, -0.2 + 1e-9, 0.1); cen = 0.5 * (edges[:-1] + edges[1:])
for sel, col, ls, mk, lab in [(upper, CU, '-', 'o', 'upper (Eos, $\\alpha$-rich)'),
                              (lower, CL, '-', 'o', 'lower (Eos, $\\alpha$-poor)'),
                              (disc, '#2ca02c', '--', 's', r'low-$\alpha$ disc ($V_{tan}>150$)')]:
    si = np.full(len(cen), np.nan); se = np.full(len(cen), np.nan)
    for i in range(len(cen)):
        b = sel & (feh >= edges[i]) & (feh < edges[i+1]); y = nfe[b]; e = nerr[b]
        if np.isfinite(y).sum() >= 10:
            si[i] = sigint(y, e)
            se[i] = np.std([sigint(*(lambda k: (y[k], e[k]))(rng.integers(0, y.size, y.size))) for _ in range(400)])
    ax[2].errorbar(cen, si, yerr=se, color=col, ls=ls, marker=mk, ms=5, lw=1.7, capsize=3, label=lab)
vals = [si_ for arr in [si] for si_ in arr]
ax[2].set_xlim(-0.92, -0.18); ax[2].set_ylim(0.03, None)
label_axes(ax[2], '[Fe/H]', r'$\sigma_{\rm [N/Fe]}$ (deconvolved) [dex]', 'N dispersion per branch')
ax[2].legend(frameon=False, fontsize=9)
fig.savefig(FIG / '01_eos_bifurcation.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_eos_bifurcation.png')
print(f'upper n={int(upper.sum())} Al={np.median(al[upper]):+.2f} N med={np.median(nfe[upper]):+.3f} sig={mad(nfe[upper]):.3f}')
print(f'lower n={int(lower.sum())} Al={np.median(al[lower]):+.2f} N med={np.median(nfe[lower]):+.3f} sig={mad(nfe[lower]):.3f}')
def rng16_84(x): return np.percentile(x, 16), np.median(x), np.percentile(x, 84)
print(f'N-RICH TAIL ([N/Fe]>{NRICH}): n={int(nrich.sum())}  ({int((nrich&upper).sum())} upper / {int((nrich&lower).sum())} lower)')
print(f'   [Fe/H]  16/50/84 = {tuple(round(v,2) for v in rng16_84(feh[nrich]))}')
print(f'   [Mg/Fe] 16/50/84 = {tuple(round(v,2) for v in rng16_84(mg[nrich]))}')
print(f'   [Al/Fe] median = {np.median(al[nrich]):+.2f}   [N/Fe] median = {np.median(nfe[nrich]):+.2f}')
