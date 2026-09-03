"""B&K22-style [N/Fe] spread (P95-P5) for the four V_tan blocks + Aurora, with the
V_tan-[Fe/H] density + labelled boxes ALWAYS on the left so the block metallicity
range (-0.8<[Fe/H]<-0.5) is explicit. Standard global-quality sample, NO per-element
N_FE_FLAG cut (B&K22 don't use it; the flag only trims one tail -> biased). Error bars
on P95-P5 are bootstrap (400 resamples, std).
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
REPO = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/eos-figures')
sys.path.insert(0, str(REPO))
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
from eos_figures.stats import hist2d
from eos_figures.plotting import density_panel, label_axes
rng = np.random.default_rng(0); c = Cuts()
FIG = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/figures_repro')
main = load_catalog('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(main, c)
feh = np.asarray(main['fe_h'], float); vphi = np.asarray(main['galvt'], float)
nfe = np.asarray(main['n_fe'], float)

BOX = (-0.8, -0.5); VLO, VHI = (-75, 75), (150, 300)
series = [('thin_al',  VLO, 'royalblue',  '-',  r'low-$\alpha$ Eos ($V_{tan}<75$)'),
          ('thin_al',  VHI, 'firebrick',  '-',  r'low-$\alpha$ disc ($V_{tan}>150$)'),
          ('thick_al', VLO, 'darkorange', '--', r'high-$\alpha$ Splash ($V_{tan}<75$)'),
          ('thick_al', VHI, 'seagreen',   '--', r'high-$\alpha$ disc ($V_{tan}>150$)')]


def p95m5(y):
    y = y[np.isfinite(y)]
    return (np.percentile(y, 95) - np.percentile(y, 5)) if y.size >= 25 else np.nan


fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3), constrained_layout=True)
# left two: density + boxes (ALWAYS, so the block extent incl. [Fe/H] is explicit)
for p, (pop, ptitle) in enumerate([('thick_al', r'high-$\alpha$'), ('thin_al', r'low-$\alpha$')]):
    P = np.asarray(m[pop], bool) & np.isfinite(feh) & np.isfinite(vphi)
    h, xe, ye = hist2d(feh[P], vphi[P], (-1.5, 0.5), (-200, 350), 70, 70, normalize='y')
    density_panel(ax[p], h, xe, ye, percentiles=(2, 98))
    ax[p].axhline(0, color='k', lw=0.6, ls=':')
    for spop, (vlo, vhi), col, ls, lab in series:
        if spop == pop:
            ax[p].add_patch(Rectangle((BOX[0], vlo), BOX[1] - BOX[0], vhi - vlo,
                                      fill=False, edgecolor=col, lw=2.2, zorder=5))
    ax[p].set_xlim(-1.5, 0.5); ax[p].set_ylim(-200, 350)
    label_axes(ax[p], '[Fe/H]', r'$V_{\rm tan}$ [km/s]', ptitle + ' sample')

# right: P95-P5 spread, ONLY within the boxed metallicity range
edges = np.arange(BOX[0], BOX[1] + 1e-9, 0.1); cen = 0.5 * (edges[:-1] + edges[1:])
allv = []
for pop, (vlo, vhi), col, ls, lab in series:
    band = np.asarray(m[pop], bool) & (vphi > vlo) & (vphi < vhi)
    sp = np.full(len(cen), np.nan); se = np.full(len(cen), np.nan)
    for i in range(len(cen)):
        y = nfe[band & (feh >= edges[i]) & (feh < edges[i + 1]) & np.isfinite(nfe)]
        if y.size >= 25:
            sp[i] = p95m5(y)
            se[i] = np.std([p95m5(y[rng.integers(0, y.size, y.size)]) for _ in range(400)])
    ax[2].errorbar(cen, sp, yerr=se, color=col, ls=ls, marker='o', ms=5, lw=1.7, capsize=3, label=lab)
    f = np.isfinite(sp); allv += list(sp[f] - np.nan_to_num(se)[f]) + list(sp[f] + np.nan_to_num(se)[f])
au = np.asarray(m['thick_al'], bool) & (feh > -1.5) & (feh < -1.0) & (vphi < 100)
av = p95m5(nfe[au])
ax[2].axhspan(av - av / np.sqrt(2 * au.sum()), av + av / np.sqrt(2 * au.sum()), color='purple', alpha=0.12)
ax[2].axhline(av, color='purple', ls=(0, (5, 2)), lw=1.3)
ax[2].text(-0.795, av, f'Aurora ($-1.5<$[Fe/H]$<-1$, $V_{{tan}}<100$)', fontsize=7, color='purple', va='bottom')
allv.append(av)
lo, hi = min(allv), max(allv); span = (hi - lo) or hi
ax[2].set_xlim(-0.82, -0.48); ax[2].set_ylim(max(0.0, lo - 0.12 * span), hi + 0.12 * span)
ax[2].text(0.5, 0.06, 'B&K22 measure = P95$-$P5;  error bars = bootstrap (400x)',
           transform=ax[2].transAxes, ha='center', fontsize=7.5, color='0.4')
label_axes(ax[2], '[Fe/H]', r'[N/Fe] P95$-$P5 [dex]', r'N spread (boxed range only)')
ax[2].legend(frameon=False, fontsize=7.5, loc='upper right')
fig.savefig(FIG / '01_eos_Nspread_bk22.png', dpi=150, bbox_inches='tight'); plt.close(fig)

print('P95-P5 within -0.8<[Fe/H]<-0.5:')
for pop, (vlo, vhi), col, ls, lab in series:
    b = np.asarray(m[pop], bool) & (vphi > vlo) & (vphi < vhi) & (feh >= BOX[0]) & (feh < BOX[1])
    print(f'  {lab:32s} n={int(np.isfinite(nfe[b]).sum()):5d}  P95-P5={p95m5(nfe[b]):.3f}')
print(f'  Aurora P95-P5={av:.3f} (n={au.sum()})')
print('wrote', FIG / '01_eos_Nspread_bk22.png')
