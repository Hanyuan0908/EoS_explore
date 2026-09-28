"""Two panels side by side, in-situ Davies halo only (accreted [Al/Fe]<-0.12 removed):
  (left)  [Mg/Fe] vs [Fe/H] scatter, stars coloured by r_apo (Eos label; [Mg/Fe]=0.3
          split line drawn as reference).
  (right) r_apo distribution of the same stars, split at [Mg/Fe]=0.3.

Data: data_repro/our_apogee_dr17_lite_ann.fits.gz. Output figures_repro/01_davies_rapo_2panel.png.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
REPO = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/eos-figures')
sys.path.insert(0, str(REPO))
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
c = Cuts()
FIG = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/figures_repro')
cat = load_catalog('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c); base = np.asarray(m['base'], bool)
feh = np.asarray(cat['fe_h'], float); mg = np.asarray(cat['mg_fe'], float); al = np.asarray(cat['al_fe'], float)
lz = np.asarray(cat['lz'], float); rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float)
ecc = (rap - rperi)/(rap + rperi)
# in-situ ([Al/Fe]>-0.12) AND metal-rich end only ([Fe/H]>-1)
halo = base & ((ecc > 0.7) | (lz < 0)) & (al > c.alfe_cut) & (feh > -1.0) & np.isfinite(rap)
XR = (-1.1, 0.6); YR = (-0.1, 0.5); VMIN, VMAX = 6, 14

fig, ax = plt.subplots(1, 2, figsize=(15.5, 6.2), constrained_layout=True)

# ---- left: scatter coloured by r_apo ----
s = halo & np.isfinite(feh) & np.isfinite(mg)
order = np.argsort(rap[s])                     # large r_apo drawn on top
xs, ys, cs = feh[s][order], mg[s][order], rap[s][order]
sc = ax[0].scatter(xs, ys, c=cs, s=9, cmap='RdYlBu_r', vmin=VMIN, vmax=VMAX,
                   edgecolors='none', alpha=0.85, rasterized=True)
ax[0].axvline(-1.0, color='red', ls='--', lw=1.5, zorder=2)
ax[0].axhline(0.3, color='k', ls=':', lw=1.4, zorder=2)          # the [Mg/Fe]=0.3 split
ax[0].text(-0.62, 0.16, 'Eos', color='k', fontsize=14, fontweight='bold', zorder=3)
ax[0].set_xlim(*XR); ax[0].set_ylim(*YR); ax[0].set_xlabel('[Fe/H]'); ax[0].set_ylabel('[Mg/Fe]')
ax[0].set_title(rf'in-situ Davies halo, coloured by $r_{{\rm apo}}$ (n={int(s.sum())})')
cb = fig.colorbar(sc, ax=ax[0], pad=0.02); cb.set_label(r'$r_{\rm apo}$ [kpc]')

# ---- right: r_apo distribution split at [Mg/Fe]=0.3 ----
hi = halo & (mg > 0.3); lo = halo & (mg <= 0.3)
edges = np.arange(0, 22.01, 1.0)
ax[1].hist(rap[halo], bins=edges, density=True, histtype='step', color='0.6', lw=1.6,
           label=f'all in-situ halo (n={int(halo.sum())})')
ax[1].hist(rap[lo], bins=edges, density=True, histtype='step', color='#1F6FB2', lw=1.8, ls='--',
           label=fr'[Mg/Fe] $\leq 0.3$ (n={int(lo.sum())}, med={np.median(rap[lo]):.1f})')
ax[1].hist(rap[hi], bins=edges, density=True, histtype='stepfilled', color='#E8112D', alpha=0.45,
           lw=2.0, edgecolor='#E8112D',
           label=fr'[Mg/Fe] $> 0.3$ (n={int(hi.sum())}, med={np.median(rap[hi]):.1f})')
ax[1].set_xlabel(r'$r_{\rm apo}$ [kpc]'); ax[1].set_ylabel('normalised density'); ax[1].set_xlim(0, 22)
ax[1].set_title(r'$r_{\rm apo}$ split at [Mg/Fe]$=0.3$ (accreted removed)')
ax[1].legend(frameon=False, fontsize=12)

fig.savefig(FIG / '01_davies_rapo_2panel.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_davies_rapo_2panel.png', ' in-situ halo n=', int(halo.sum()))
