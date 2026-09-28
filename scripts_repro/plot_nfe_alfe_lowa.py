"""[N/Fe] vs [Al/Fe] for the low-alpha (in-situ) population, WITHOUT the
[Al/Fe] > -0.12 in-situ floor (mask 'thin', not 'thin_al'), in the obs_nfe_pops
house style:
  top    : column-normalised (per [Al/Fe]) [N/Fe] density (Greys) + P5/P95 tracks (red)
  bottom : same plane coloured by median V_phi (RdYlBu_r)
A vertical dashed line marks where the [Al/Fe] = -0.12 floor would sit, so the low-Al
(potentially accreted) tail that the floor removes is visible.

Data: data_repro/our_apogee_dr17_lite_ann.fits.gz. Output figures_repro/01_nfe_alfe_lowa.png.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

REPO = '/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore'
sys.path.insert(0, REPO + '/eos-figures')
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
from eos_figures.stats import hist2d, stat2d, log_image, finite_percentile
from eos_figures.figures import _idl_low_density_mask

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 14, 'axes.labelsize': 17,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})
FIG = Path(REPO + '/figures_repro')
c = Cuts()
cat = load_catalog(REPO + '/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c)
al = np.asarray(cat['al_fe'], float)
nfe = np.asarray(cat['n_fe'], float)
vt = np.asarray(cat['galvt'], float)

lowa = np.asarray(m['thin'], bool)          # low-alpha WITHOUT the [Al/Fe]>-0.12 floor
XR = (-0.55, 0.45)                           # [Al/Fe]
YR = (-0.5, 0.7)                             # [N/Fe]
NX, NY = 28, 28
RED = '#E8112D'; FLOOR = c.alfe_cut          # -0.12


def pctl_tracks(mask):
    edges = np.linspace(XR[0], XR[1], NX + 1); cen = 0.5 * (edges[:-1] + edges[1:])
    p5 = np.full(NX, np.nan); p95 = np.full(NX, np.nan)
    for i in range(NX):
        yy = nfe[mask & (al >= edges[i]) & (al < edges[i + 1]) & np.isfinite(nfe)]
        if yy.size >= 15:
            p5[i], p95[i] = np.percentile(yy, [5, 95])
    return cen, p5, p95


fig, ax = plt.subplots(2, 1, figsize=(6.4, 8.4), sharex=True, constrained_layout=True)

# --- top: column-normalised density + P5/P95 ---
h, xe, ye = hist2d(al[lowa], nfe[lowa], XR, YR, NX, NY, normalize='x')
lowmask = _idl_low_density_mask(h, c.perc2, c.white_lim)
im = log_image(h); dvmin, dvmax = finite_percentile(im[np.isfinite(im)], c.perc2)
im_top = ax[0].imshow(im.T, origin='lower', extent=[xe[0], xe[-1], ye[0], ye[-1]],
                      aspect='auto', interpolation='nearest', cmap='Greys', vmin=dvmin, vmax=dvmax)
im_top.set_rasterized(True)
cen, p5, p95 = pctl_tracks(lowa)
ax[0].plot(cen, p95, color=RED, lw=2.4, zorder=6, label='P95 / P5')
ax[0].plot(cen, p5, color=RED, lw=2.4, zorder=6)
ax[0].axvline(FLOOR, color='k', ls='--', lw=1.2, zorder=5)
ax[0].text(FLOOR - 0.01, YR[1] - 0.06, r'[Al/Fe]$=-0.12$', rotation=90, va='top', ha='right', fontsize=11, color='0.25')
ax[0].set_ylabel('[N/Fe]'); ax[0].set_ylim(*YR)
cb0 = fig.colorbar(im_top, ax=ax[0], location='right', pad=0.02, aspect=26)
cb0.set_label(r'$\log_{10}$ (norm. density)'); cb0.ax.tick_params(length=3)
ax[0].set_title(r'low-$\alpha$ (in-situ), no [Al/Fe] floor', fontsize=15)

# --- bottom: median V_phi ---
vmask = lowa & np.isfinite(vt) & (vt >= c.vtanr[0]) & (vt <= c.vtanr[1])
med, _, _ = stat2d(al[vmask], nfe[vmask], vt[vmask], XR, YR, NX, NY)
h_med, _, _ = hist2d(al[vmask], nfe[vmask], XR, YR, NX, NY)
med = np.nan_to_num(med, nan=0.0); med[h_med <= 2] = 0.0
img = np.array(med, float); img[lowmask] = np.nan
im_bot = ax[1].imshow(img.T, origin='lower', extent=[xe[0], xe[-1], ye[0], ye[-1]],
                      aspect='auto', interpolation='nearest', cmap='RdYlBu_r',
                      vmin=c.mm_vtan[0], vmax=c.mm_vtan[1])
im_bot.set_rasterized(True)
ax[1].axvline(FLOOR, color='k', ls='--', lw=1.2, zorder=5)
ax[1].set_xlabel('[Al/Fe]'); ax[1].set_ylabel('[N/Fe]')
ax[1].set_xlim(*XR); ax[1].set_ylim(*YR)
cb1 = fig.colorbar(im_bot, ax=ax[1], location='right', pad=0.02, aspect=26)
cb1.set_label(r'$V_\phi$ [km/s]'); cb1.ax.tick_params(length=3)

fig.savefig(FIG / '01_nfe_alfe_lowa.png', bbox_inches='tight')
print('wrote', FIG / '01_nfe_alfe_lowa.png', ' low-a (thin) n=', int(lowa.sum()),
      ' below floor:', int((lowa & (al < FLOOR)).sum()))
