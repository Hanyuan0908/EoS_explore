"""Exploratory (NOT for publication): low-alpha (thin_al) population in the
[Fe/H]-V_tan plane, three panels.
  Left   : number counts per pixel (cmasher amber, log scale).
  Middle : mean apocentric radius  r_apo  per pixel.
  Right  : mean pericentric radius r_peri per pixel.
Pixel setup (bins=70x70, min_count=1, ranges) copied from eos_figures.plot_vphi_pixels.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cmasher as cmr
REPO = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/eos-figures')
sys.path.insert(0, str(REPO))
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
from eos_figures.stats import hist2d, stat2d
from eos_figures.plotting import value_panel, label_axes
c = Cuts()
FIG = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/figures_repro')
cat = load_catalog('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c)
feh = np.asarray(cat['fe_h'], float); vphi = np.asarray(cat['galvt'], float)
rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float)
mask = np.asarray(m['thin_al'], bool)

XR, YR = c.fehr_plot, c.vphir_plot2; NB = 70; MINCOUNT = 1   # pixel setup from plot_vphi_pixels
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True, constrained_layout=True)

# --- left: number counts (cmasher amber) ---
good = mask & np.isfinite(feh) & np.isfinite(vphi)
h = ax[0].hist2d(feh[good], vphi[good], bins=[NB, NB], range=[XR, YR], cmap=cmr.amber, norm=LogNorm())
fig.colorbar(h[3], ax=ax[0], pad=0.02).set_label('count')
label_axes(ax[0], '[Fe/H]', r'$V_{\rm tan}$ [km/s]', 'counts')

# --- middle / right: mean r_apo and r_peri per pixel (as in plot_vphi_pixels) ---
for axis, values, mm, lab, ttl in [(ax[1], rap, c.mm_rapo, r'$r_{\rm apo}$ [kpc]', r'mean $r_{\rm apo}$'),
                                   (ax[2], rperi, c.mm_rperi, r'$r_{\rm peri}$ [kpc]', r'mean $r_{\rm peri}$')]:
    mn, xe, ye = stat2d(feh[mask], vphi[mask], values[mask], XR, YR, NB, NB, statistic='mean')
    cnt, _, _ = hist2d(feh[mask], vphi[mask], XR, YR, NB, NB)
    im = value_panel(axis, mn, xe, ye, mm[0], mm[1], mask=cnt < MINCOUNT, cmap='RdYlBu_r', colorbar_label=lab)
    im.set_rasterized(True)
    axis.axhline(0, color='k', ls='--', lw=0.8)
    axis.set_xlim(*XR); axis.set_ylim(*YR)
    label_axes(axis, '[Fe/H]', '', ttl)

fig.savefig(FIG / '01_lowa_vphi_feh_ecc.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_lowa_vphi_feh_ecc.png')
