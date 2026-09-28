"""Two extra versions of the Davies-Fig2 right panel ([Mg/Fe] vs [Fe/H], Davies halo
e>0.7 | Lz<0):
  (1) inner-galaxy counterpart of the R>5 figure: R < 5 kpc log-density
      -> 01_davies_fig2_mg_r5in.png
  (2) full halo coloured by mean apocentric radius r_apo per bin (clean value map)
      -> 01_davies_fig2_mg_rapo.png

R = galr from the Mac-only AstroNN VAC (matched by APOGEE_ID); r_apo = rap from the
in-repo lite cache. Red dashed line at [Fe/H]=-1.1; 'Eos' label as in the original.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic_2d
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

ann = fits.open('/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE/apogee_astroNN-DR17.fits')[1].data
def norm(a): return np.array([(s.decode() if isinstance(s, bytes) else str(s)).strip() for s in np.asarray(a)])
nid = norm(ann['APOGEE_ID']); o = np.argsort(nid); nid_s = nid[o]
p = np.clip(np.searchsorted(nid_s, aid), 0, len(nid_s)-1); ok = nid_s[p] == aid; src = o[p]
R = np.where(ok, np.asarray(ann['galr'], float)[src], np.nan)

XR = (-2.1, 0.6); YR = (-0.1, 0.5); BINS = [75, 60]

# ---- (1) inner R < 5 kpc density ----
sin = halo & (R < 5) & np.isfinite(feh) & np.isfinite(mg)
fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
ax.hist2d(feh[sin], mg[sin], bins=BINS, range=[XR, YR], cmap='Greys', norm=LogNorm())
ax.axvline(-1.1, color='red', ls='--', lw=1.5, zorder=2)
ax.text(-0.62, 0.16, 'Eos', color='red', fontsize=14, fontweight='bold', zorder=3)
ax.set_xlim(*XR); ax.set_ylim(*YR); ax.set_xlabel('[Fe/H]'); ax.set_ylabel('[Mg/Fe]')
ax.set_title(rf'halo (Davies $e>0.7\,|\,L_z<0$) & $R<5$ kpc, n={int(sin.sum())}')
fig.savefig(FIG / '01_davies_fig2_mg_r5in.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_davies_fig2_mg_r5in.png', ' n=', int(sin.sum()))

# ---- (2) full halo coloured by mean r_apo (larger pixels: coarser bins) ----
sv = halo & np.isfinite(feh) & np.isfinite(mg) & np.isfinite(rap)
NMIN = 1        # colour every occupied bin (match the density-plot coverage)
VBINS = [42, 34]   # coarser than the density grid -> larger pixels
vmin, vmax = 6, 14
me = binned_statistic_2d(feh[sv], mg[sv], rap[sv], statistic='mean', bins=VBINS, range=[XR, YR]).statistic
cnt = binned_statistic_2d(feh[sv], mg[sv], None, statistic='count', bins=VBINS, range=[XR, YR]).statistic
mj = np.where(cnt >= NMIN, me, np.nan)
fig2, ax2 = plt.subplots(figsize=(7.8, 6.2), constrained_layout=True)
im = ax2.imshow(mj.T, origin='lower', extent=[*XR, *YR], aspect='auto', cmap='RdYlBu_r',
                vmin=vmin, vmax=vmax, zorder=0)
ax2.axvline(-1.1, color='red', ls='--', lw=1.5, zorder=2)
ax2.text(-0.62, 0.16, 'Eos', color='k', fontsize=14, fontweight='bold', zorder=3)
ax2.set_xlim(*XR); ax2.set_ylim(*YR); ax2.set_xlabel('[Fe/H]'); ax2.set_ylabel('[Mg/Fe]')
ax2.set_title(rf'Davies halo, coloured by mean $r_{{\rm apo}}$ (n={int(sv.sum())})')
cb = fig2.colorbar(im, ax=ax2, pad=0.02); cb.set_label(r'mean $r_{\rm apo}$ [kpc]')
fig2.savefig(FIG / '01_davies_fig2_mg_rapo.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_davies_fig2_mg_rapo.png', ' n=', int(sv.sum()))

# ---- (3) scatter of individual stars coloured by r_apo (high-r_apo drawn on top) ----
order = np.argsort(rap[sv])           # ascending -> large r_apo plotted last (on top)
xs, ys, cs = feh[sv][order], mg[sv][order], rap[sv][order]
fig3, ax3 = plt.subplots(figsize=(7.8, 6.2), constrained_layout=True)
sc = ax3.scatter(xs, ys, c=cs, s=9, cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
                 edgecolors='none', alpha=0.85, rasterized=True)
ax3.axvline(-1.1, color='red', ls='--', lw=1.5, zorder=2)
ax3.text(-0.62, 0.16, 'Eos', color='k', fontsize=14, fontweight='bold', zorder=3)
ax3.set_xlim(*XR); ax3.set_ylim(*YR); ax3.set_xlabel('[Fe/H]'); ax3.set_ylabel('[Mg/Fe]')
ax3.set_title(rf'Davies halo, stars coloured by $r_{{\rm apo}}$ (n={int(sv.sum())})')
cb3 = fig3.colorbar(sc, ax=ax3, pad=0.02); cb3.set_label(r'$r_{\rm apo}$ [kpc]')
fig3.savefig(FIG / '01_davies_fig2_mg_rapo_scatter.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_davies_fig2_mg_rapo_scatter.png', ' n=', int(sv.sum()))
