"""Publication figure (observational): nitrogen dispersion, Eos vs low-alpha disc,
with the high-alpha (Splash) benchmark.

  (a) high-alpha sample in [Fe/H]-V_phi (row-normalised density), with a
      low-V_phi Splash box and a disc box over -0.8<[Fe/H]<-0.5.
  (b) low-alpha sample, same plane, with the low-V_phi Eos box and disc box.
  (c) robust sigma_[N/Fe] (1.48xMAD) vs [Fe/H] for the four bands, coloured to
      match the boxes (low-alpha solid, high-alpha dashed); the Aurora level is
      the purple band. The low-V_phi N excess appears in low-alpha (Eos) but not
      in high-alpha (Splash) -- it is not a generic heating effect.

Derived from scripts_repro/build_nb.py::disp_figure. Changes for publication:
V_tan -> V_phi throughout; the two grey matched-Delta-sigma annotations removed;
right-panel lower y-limit set to 0.05; legend given headroom above the Aurora line.

Data: data_repro/our_apogee_dr17_lite_ann.fits.gz (in-repo, portable).
Writes Fig_paper/obs_ndispersion.pdf and .png.
"""
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = '/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore'
sys.path.insert(0, REPO + '/eos-figures')
# imports first, then our own rcParams below so the publication style wins.
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
from eos_figures.stats import hist2d
from eos_figures.plotting import density_panel, label_axes

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 15, 'axes.labelsize': 19, 'axes.titlesize': 18,
    'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 13.5,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})

OUT = REPO + '/Fig_paper'
os.makedirs(OUT, exist_ok=True)
c = Cuts()
cat = load_catalog(REPO + '/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c)

feh = np.asarray(cat['fe_h'], float)
vphi = np.asarray(cat['galvt'], float)
y = np.asarray(cat['n_fe'], float)

FEHR_BOX = (-0.8, -0.5)
VLO, VHI = (-75, 75), (150, 300)
series = [('thin_al',  VLO, 'royalblue',  '-',  r'low-$\alpha$ Eos ($V_\phi<75$)'),
          ('thin_al',  VHI, 'firebrick',  '-',  r'low-$\alpha$ disc ($V_\phi>150$)'),
          ('thick_al', VLO, 'darkorange', '--', r'high-$\alpha$ Splash ($V_\phi<75$)'),
          ('thick_al', VHI, 'seagreen',   '--', r'high-$\alpha$ disc ($V_\phi>150$)')]


def running_sigma(band, x, yy_all, xr, nb=3, minn=10):
    edges = np.linspace(*xr, nb + 1); cen = 0.5 * (edges[:-1] + edges[1:])
    sig = np.full(nb, np.nan); err = np.full(nb, np.nan)
    for i in range(nb):
        b = band & (x >= edges[i]) & (x < edges[i+1]) & np.isfinite(yy_all); yy = yy_all[b]
        if yy.size >= minn:
            s = 1.4826 * np.median(np.abs(yy - np.median(yy)))
            sig[i] = s; err[i] = s / np.sqrt(2 * yy.size)
    return cen, sig, err


fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3), constrained_layout=True)

# (a),(b) density in [Fe/H]-V_phi with the selection boxes
for p, (pop, ptitle) in enumerate([('thick_al', r'high-$\alpha$'), ('thin_al', r'low-$\alpha$')]):
    P = np.asarray(m[pop], bool) & np.isfinite(feh) & np.isfinite(vphi)
    h, xe, ye = hist2d(feh[P], vphi[P], (-1.5, 0.5), (-200, 350), 70, 70, normalize='y')
    im = density_panel(ax[p], h, xe, ye, percentiles=(2, 98)); im.set_rasterized(True)
    ax[p].axhline(0, color='k', lw=0.6, ls=':')
    for spop, (vlo, vhi), col, ls, lab in series:
        if spop == pop:
            ax[p].add_patch(Rectangle((FEHR_BOX[0], vlo), FEHR_BOX[1]-FEHR_BOX[0], vhi-vlo,
                                      fill=False, edgecolor=col, lw=2.2, zorder=5))
    ax[p].set_xlim(-1.5, 0.5); ax[p].set_ylim(-200, 350)
    label_axes(ax[p], '[Fe/H]', r'$V_\phi$ [km/s]', ptitle + ' sample')

# (c) robust sigma_[N/Fe] vs [Fe/H]
for pop, (vlo, vhi), col, ls, lab in series:
    band = np.asarray(m[pop], bool) & (vphi > vlo) & (vphi < vhi)
    cen, sig, err = running_sigma(band, feh, y, FEHR_BOX)
    ax[2].errorbar(cen, sig, yerr=err, color=col, ls=ls, marker='o', ms=5, lw=1.6, capsize=3, label=lab)
# Aurora benchmark band
hv, he = 0.149, 0.004
ax[2].axhspan(hv - he, hv + he, color='purple', alpha=0.12, zorder=0)
ax[2].axhline(hv, color='purple', ls=(0, (5, 2)), lw=1.3, zorder=1)
ax[2].text(-0.815, hv + he, r'Aurora ($-1.5<$[Fe/H]$<-1$, $V_\phi<100$)',
           fontsize=9, color='purple', va='bottom', ha='left')
ax[2].set_xlim(-0.82, -0.48); ax[2].set_ylim(0.05, 0.19)    # hard cap 0.19; blank band 0.16-0.19 holds the legend
ax[2].set_yticks([0.06, 0.08, 0.10, 0.12, 0.14, 0.16])       # no ticks in the legend band (all data < 0.155)
label_axes(ax[2], '[Fe/H]', r'$\sigma_{\rm [N/Fe]}$ [dex]', r'N dispersion: low- vs high-$V_\phi$')
ax[2].legend(frameon=False, fontsize=11, loc='upper right', ncol=2, columnspacing=1.2, handlelength=1.6)

for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/obs_ndispersion.{ext}', bbox_inches='tight')
print('wrote', OUT + '/obs_ndispersion.{pdf,png}')
