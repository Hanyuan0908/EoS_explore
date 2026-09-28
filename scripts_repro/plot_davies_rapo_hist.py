"""r_apo distribution of Davies-halo (e>0.7 | Lz<0) stars with [Mg/Fe] > 0.3
(the high-alpha clump region of the [Mg/Fe]-[Fe/H] plane). The whole halo and the
[Mg/Fe] < 0.3 complement are shown as light references.

Data: data_repro/our_apogee_dr17_lite_ann.fits.gz. Output figures_repro/01_davies_rapo_hist.png.
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
# in-situ only: drop accreted stars with [Al/Fe] < -0.12
halo = base & ((ecc > 0.7) | (lz < 0)) & (al > c.alfe_cut) & np.isfinite(rap)

hi = halo & (mg > 0.3)          # high-alpha clump region
lo = halo & (mg <= 0.3)         # complement (low-alpha / Eos / accreted)
edges = np.arange(0, 22.01, 1.0)

fig, ax = plt.subplots(figsize=(8.2, 5.6), constrained_layout=True)
ax.hist(rap[halo], bins=edges, density=True, histtype='step', color='0.6', lw=1.6, label=f'all halo (n={int(halo.sum())})')
ax.hist(rap[lo], bins=edges, density=True, histtype='step', color='#1F6FB2', lw=1.8, ls='--',
        label=fr'[Mg/Fe] $\leq 0.3$ (n={int(lo.sum())}, med={np.median(rap[lo]):.1f})')
ax.hist(rap[hi], bins=edges, density=True, histtype='stepfilled', color='#E8112D', alpha=0.45, lw=2.0, edgecolor='#E8112D',
        label=fr'[Mg/Fe] $> 0.3$ (n={int(hi.sum())}, med={np.median(rap[hi]):.1f})')
ax.axvline(np.median(rap[hi]), color='#E8112D', ls=':', lw=1.5)
ax.set_xlabel(r'$r_{\rm apo}$ [kpc]'); ax.set_ylabel('normalised density')
ax.set_xlim(0, 22)
ax.set_title(r'$r_{\rm apo}$ of Davies halo stars, split at [Mg/Fe]$=0.3$')
ax.legend(frameon=False, fontsize=12)
fig.savefig(FIG / '01_davies_rapo_hist.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_davies_rapo_hist.png')
for lab, s in [('mg>0.3', hi), ('mg<=0.3', lo), ('all halo', halo)]:
    r = rap[s]
    print(f'  {lab:9s}: n={r.size:5d}  median r_apo={np.median(r):.2f}  16-84=[{np.percentile(r,16):.1f},{np.percentile(r,84):.1f}]  frac(r_apo>10)={np.mean(r>10):.2f}')
