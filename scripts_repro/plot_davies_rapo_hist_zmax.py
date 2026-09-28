"""r_apo distribution of the in-situ Davies halo ([Al/Fe]>-0.12, [Fe/H]>-1), split
at [Mg/Fe]=0.3 (high- vs low-alpha), with the additional cut z_max > 2 kpc (off-plane
only -- removes the bar/bulge). Companion to plot_davies_rapo_2panel.py with z_max
added. Output figures_repro/01_davies_rapo_hist_zmax2.png.
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
zmax = np.asarray(cat['zmax'], float)
ecc = (rap - rperi)/(rap + rperi)
# in-situ Davies halo, [Fe/H]>-1, off-plane (z_max>2)
halo = base & ((ecc > 0.7) | (lz < 0)) & (al > c.alfe_cut) & (feh > -1.0) & (zmax > 2.0) & np.isfinite(rap)
hi = halo & (mg > 0.3); lo = halo & (mg <= 0.3)
edges = np.arange(0, 22.01, 1.0)

fig, ax = plt.subplots(figsize=(8.2, 5.6), constrained_layout=True)
ax.hist(rap[halo], bins=edges, density=True, histtype='step', color='0.6', lw=1.6,
        label=f'all (n={int(halo.sum())})')
ax.hist(rap[lo], bins=edges, density=True, histtype='step', color='#1F6FB2', lw=1.8, ls='--',
        label=fr'low-$\alpha$ [Mg/Fe]$\leq0.3$ (n={int(lo.sum())}, med={np.median(rap[lo]):.1f})')
ax.hist(rap[hi], bins=edges, density=True, histtype='stepfilled', color='#E8112D', alpha=0.45,
        lw=2.0, edgecolor='#E8112D',
        label=fr'high-$\alpha$ [Mg/Fe]$>0.3$ (n={int(hi.sum())}, med={np.median(rap[hi]):.1f})')
ax.set_xlabel(r'$r_{\rm apo}$ [kpc]'); ax.set_ylabel('normalised density'); ax.set_xlim(0, 22)
ax.set_title(r'$r_{\rm apo}$ split at [Mg/Fe]$=0.3$  (in-situ, [Fe/H]$>-1$, $z_{\rm max}>2$ kpc)')
ax.legend(frameon=False, fontsize=12)
fig.savefig(FIG / '01_davies_rapo_hist_zmax2.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_davies_rapo_hist_zmax2.png')
for lab, s in [('high-a (mg>0.3)', hi), ('low-a (mg<=0.3)', lo), ('all', halo)]:
    r = rap[s]
    print(f'  {lab:16s}: n={r.size:4d}  median r_apo={np.median(r):.2f}  frac(>10)={np.mean(r>10):.2f}')
