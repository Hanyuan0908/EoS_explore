"""Davies+2025 Fig 2 style, LAMOST version: the halo (Davies cut base & (ecc>0.7 | Lz<0))
in [Al/Fe]-[Fe/H] (left) and [Mg/Fe]-[Fe/H] (right) as log-density. Question BEFORE ages:
are the two distinct Eos components (the two clumps around [Fe/H]~-0.6 in [Mg/Fe]) even
differentiable in LAMOST as they are in APOGEE?
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
cat = load_catalog('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_lamost_subgiant_ddpayne.fits.gz')
m = make_masks(cat, c); base = np.asarray(m['base'], bool)
feh = np.asarray(cat['fe_h'], float); mg = np.asarray(cat['mg_fe'], float); al = np.asarray(cat['al_fe'], float)
lz = np.asarray(cat['lz'], float); rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float)
ecc = (rap - rperi) / (rap + rperi)
halo = base & ((ecc > 0.7) | (lz < 0))
n = int((halo & np.isfinite(feh)).sum())

def panel(ax, y, yr, ylab):
    s = halo & np.isfinite(feh) & np.isfinite(y)
    h, xe, ye = np.histogram2d(feh[s], y[s], bins=[70, 55], range=[(-2.1, 0.6), yr])
    him = np.full_like(h, np.nan); him[h > 0] = np.log10(h[h > 0])
    ax.imshow(him.T, origin='lower', extent=[-2.1, 0.6, *yr], aspect='auto', cmap='Greys',
              vmin=np.nanpercentile(him, 2), vmax=np.nanpercentile(him, 99), zorder=0)
    ax.axvline(-1.1, color='red', ls='--', lw=1.5, zorder=2)
    ax.set_xlim(-2.1, 0.6); ax.set_ylim(*yr)
    ax.set_xlabel('[Fe/H]'); ax.set_ylabel(ylab)
    ax.set_title(f'LAMOST halo (Davies $e>0.7|L_z<0$), n={n}')

fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
panel(ax[0], al, (-0.45, 0.45), '[Al/Fe]')
panel(ax[1], mg, (-0.1, 0.5), '[Mg/Fe]')
ax[1].text(-0.62, 0.16, 'Eos', color='red', fontsize=13, fontweight='bold', zorder=3)
fig.savefig(FIG / '01_davies_fig2_lamost.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_davies_fig2_lamost.png', 'n_halo=', n)
