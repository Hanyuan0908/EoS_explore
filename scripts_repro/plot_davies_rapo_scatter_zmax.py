"""Same as the Davies-halo [Mg/Fe]-[Fe/H] scatter coloured by r_apo, but with the
additional cut z_max > 2 kpc -- keeping only vertically-extended (off-plane) stars,
which removes the in-plane bar/bulge and thin-disc populations.

z_max and r_apo are both in the in-repo lite cache (AstroNN orbital parameters).
Output figures_repro/01_davies_rapo_scatter_zmax2.png.
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
feh = np.asarray(cat['fe_h'], float); mg = np.asarray(cat['mg_fe'], float)
lz = np.asarray(cat['lz'], float); rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float)
zmax = np.asarray(cat['zmax'], float)
ecc = (rap - rperi)/(rap + rperi)
halo = base & ((ecc > 0.7) | (lz < 0)) & np.isfinite(rap) & np.isfinite(mg) & np.isfinite(feh)
sel = halo & np.isfinite(zmax) & (zmax > 2.0)          # off-plane only
print('halo n=', int(halo.sum()), ' halo & zmax>2 n=', int(sel.sum()))

XR = (-2.1, 0.6); YR = (-0.1, 0.5); VMIN, VMAX = 6, 14
order = np.argsort(rap[sel])                            # large r_apo on top
xs, ys, cs = feh[sel][order], mg[sel][order], rap[sel][order]
fig, ax = plt.subplots(figsize=(7.8, 6.2), constrained_layout=True)
sc = ax.scatter(xs, ys, c=cs, s=9, cmap='RdYlBu_r', vmin=VMIN, vmax=VMAX,
                edgecolors='none', alpha=0.85, rasterized=True)
ax.axvline(-1.1, color='red', ls='--', lw=1.5, zorder=2)
ax.text(-0.62, 0.16, 'Eos', color='k', fontsize=14, fontweight='bold', zorder=3)
ax.set_xlim(*XR); ax.set_ylim(*YR); ax.set_xlabel('[Fe/H]'); ax.set_ylabel('[Mg/Fe]')
ax.set_title(rf'Davies halo & $z_{{\rm max}}>2$ kpc, coloured by $r_{{\rm apo}}$ (n={int(sel.sum())})')
cb = fig.colorbar(sc, ax=ax, pad=0.02); cb.set_label(r'$r_{\rm apo}$ [kpc]')
fig.savefig(FIG / '01_davies_rapo_scatter_zmax2.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_davies_rapo_scatter_zmax2.png')
