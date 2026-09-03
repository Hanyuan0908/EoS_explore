"""Publication figure (observational): the low-alpha (in-situ) population in the
[Fe/H]-V_tan plane, three pixel panels (bins=70x70, min_count=1 -- the pixel setup
of eos_figures.plot_vphi_pixels).

  (a) number counts per pixel               (cmasher amber, log scale)
  (b) mean apocentric radius   r_apo        (RdYlBu_r)
  (c) mean pericentric radius  r_peri       (RdYlBu_r)

The disc ridge (V_tan~200) is near-circular (small r_apo, moderate r_peri); the
slow / non-rotating Eos-halo foot below ~100 km/s is on plunging eccentric orbits
(large r_apo, tiny r_peri), most extreme at the metal-poor edge.

Data: data_repro/our_apogee_dr17_lite_ann.fits.gz (in-repo, portable).
Writes Fig_paper/obs_lowa_vtan_pixels.pdf and .png.
"""
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cmasher as cmr

REPO = '/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore'
sys.path.insert(0, REPO + '/eos-figures')
# import only stats/data/config (NOT plotting) so the reference rcParams are not
# applied and our own publication style below wins.
from eos_figures.stats import hist2d, stat2d
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts

OUT = REPO + '/Fig_paper'
os.makedirs(OUT, exist_ok=True)

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 16, 'axes.labelsize': 21,
    'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 13,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})

c = Cuts()
cat = load_catalog(REPO + '/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c)
feh = np.asarray(cat['fe_h'], float); vphi = np.asarray(cat['galvt'], float)
rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float)
sel = np.asarray(m['thin_al'], bool)
XR, YR = c.fehr_plot, (-250, 370); NB = 70; MINCOUNT = 1   # pixel setup from plot_vphi_pixels; y extended for cbar space


def inset_cbar(ax, im, label, ticks=None):
    cax = ax.inset_axes([0.09, 0.095, 0.86, 0.05])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label(label, fontsize=19)
    if ticks is not None:
        cb.set_ticks(ticks)
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('bottom')
    cb.ax.tick_params(labelsize=16, length=3)
    cb.outline.set_linewidth(0.7)
    return cb


fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True, constrained_layout=True)

# (a) counts
good = sel & np.isfinite(feh) & np.isfinite(vphi)
h = ax[0].hist2d(feh[good], vphi[good], bins=[NB, NB], range=[XR, YR], cmap=cmr.amber, norm=LogNorm())
h[3].set_rasterized(True)
inset_cbar(ax[0], h[3], 'count')

# (b),(c) mean r_apo / r_peri per pixel
for axis, values, mm, label in [(ax[1], rap, c.mm_rapo, r'$r_{\rm apo}$ [kpc]'),
                                (ax[2], rperi, c.mm_rperi, r'$r_{\rm peri}$ [kpc]')]:
    mn, xe, ye = stat2d(feh[sel], vphi[sel], values[sel], XR, YR, NB, NB, statistic='mean')
    cnt, _, _ = hist2d(feh[sel], vphi[sel], XR, YR, NB, NB)
    img = np.array(mn, float); img[cnt < MINCOUNT] = np.nan
    im = axis.imshow(img.T, origin='lower', extent=[xe[0], xe[-1], ye[0], ye[-1]], aspect='auto',
                     interpolation='nearest', cmap='RdYlBu_r', vmin=mm[0], vmax=mm[1])
    im.set_rasterized(True)
    inset_cbar(axis, im, label, ticks=np.linspace(mm[0], mm[1], 5))

for a, tag in zip(ax, ['(a)', '(b)', '(c)']):
    a.axhline(0, color='k', ls='--', lw=0.8)
    a.set_xlim(*XR); a.set_ylim(*YR)
    a.set_xticks([-1.5, -1.0, -0.5, 0.0, 0.5])
    a.set_xlabel('[Fe/H]')
    a.text(0.045, 0.955, tag, transform=a.transAxes, fontsize=16, fontweight='bold',
           va='top', ha='left', bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.5))
ax[0].set_ylabel(r'$V_\phi$ [km/s]')

for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/obs_lowa_vtan_pixels.{ext}', bbox_inches='tight')
print('wrote', OUT + '/obs_lowa_vtan_pixels.{pdf,png}')
