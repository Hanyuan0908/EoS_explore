"""Right panel of the Davies-Fig2 reproduction: [Mg/Fe] vs [Fe/H] log-density for the
Davies halo (e>0.7 | Lz<0), with the additional Galactocentric-radius cut R > 5 kpc.

R = galr from the Mac-only AstroNN VAC (apogee_astroNN-DR17.fits), matched by APOGEE_ID.
The R>5 cut removes the inner-galaxy/bulge clump (the metal-rich blob at [Fe/H]~0.35).
Red dashed line at [Fe/H]=-1.1; 'Eos' label as in the original.

Output figures_repro/01_davies_fig2_mg_r5.png.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
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
ecc = (rap - rperi)/(rap + rperi); aid = np.asarray(cat['apogee_id'])
halo = base & ((ecc > 0.7) | (lz < 0))

# galr from the Mac-only AstroNN VAC (matched by APOGEE_ID)
ann = fits.open('/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE/apogee_astroNN-DR17.fits')[1].data
def norm(a): return np.array([(s.decode() if isinstance(s, bytes) else str(s)).strip() for s in np.asarray(a)])
nid = norm(ann['APOGEE_ID']); o = np.argsort(nid); nid_s = nid[o]
p = np.clip(np.searchsorted(nid_s, aid), 0, len(nid_s)-1); ok = nid_s[p] == aid; src = o[p]
R = np.where(ok, np.asarray(ann['galr'], float)[src], np.nan)

sel = halo & (R > 5) & np.isfinite(feh) & np.isfinite(mg)
print('halo n=', int(halo.sum()), ' halo & R>5 n=', int(sel.sum()))

fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
h = ax.hist2d(feh[sel], mg[sel], bins=[75, 60], range=[(-2.1, 0.6), (-0.1, 0.5)],
              cmap='Greys', norm=LogNorm())
ax.axvline(-1.1, color='red', ls='--', lw=1.5, zorder=2)
ax.text(-0.62, 0.16, 'Eos', color='red', fontsize=14, fontweight='bold', zorder=3)
ax.set_xlim(-2.1, 0.6); ax.set_ylim(-0.1, 0.5)
ax.set_xlabel('[Fe/H]'); ax.set_ylabel('[Mg/Fe]')
ax.set_title(rf'halo (Davies $e>0.7\,|\,L_z<0$) & $R>5$ kpc, n={int(sel.sum())}')
fig.savefig(FIG / '01_davies_fig2_mg_r5.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_davies_fig2_mg_r5.png')
