"""Publication figure (observational): [N/Fe]-[Fe/H] for the accreted + two in-situ
populations (accreted / high-alpha / low-alpha), 2x3.

Same style as fig_paper_nfe_pops.py (obs_nfe_pops), with the ACCRETED column added
on the left:
  top row    : column-normalised [N/Fe]-[Fe/H] density (Greys, shared stretch), with
               the 5th and 95th percentile tracks of [N/Fe] vs [Fe/H] (both red).
  bottom row : the same planes coloured by median V_phi.
Colourbars sit at the far right, one per row. Panels use equal physical aspect.

The accreted panel carries our extra chemical outlier rejection (N/Fe figure only):
accreted stars above the line (-1.5, 0.6)->(-0.5, 0.42) or below (-1.5, -0.2)->
(-1.0, -0.4) are removed. These are chemical outliers --
the accreted sample already has |Lz| < 500 and a finite V_phi for every star. The
two in-situ panels are identical to obs_nfe_pops.

Data: data_repro/our_apogee_dr17_lite_ann.fits.gz (in-repo, portable).
`python fig_paper_nfe_pops_3pop.py [n_fe|c_fe]`  (default n_fe)
Writes Fig_paper/obs_nfe_pops_3pop.pdf and .png (obs_cfe_pops_3pop for c_fe).
"""
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

REPO = '/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore'
sys.path.insert(0, REPO + '/eos-figures')
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
from eos_figures.stats import hist2d, stat2d, log_image, finite_percentile
from eos_figures.figures import _idl_low_density_mask

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 14, 'axes.labelsize': 16,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12.5,
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

COL = sys.argv[1] if len(sys.argv) > 1 else 'n_fe'
PARAMS = {'n_fe': ((-0.5, 0.7), '[N/Fe]', 'obs_nfe_pops_3pop'),
          'c_fe': ((-0.6, 0.4), '[C/Fe]', 'obs_cfe_pops_3pop')}
YR, YLAB, OUTNAME = PARAMS[COL]
XR = (-1.5, c.fehr[1])
NFEH = int(round((XR[1] - XR[0]) / ((c.fehr[1] - c.fehr[0]) / c.nfeh)))
RED = '#E8112D'
GREEN = '#2E7D32'

y = np.asarray(cat[COL], float)
feh = np.asarray(cat['fe_h'], float)
vt = np.asarray(cat['galvt'], float)

# accreted chemical outlier cut (N/Fe figure only) -- see docstring
nfe_all = np.asarray(cat['n_fe'], float)
N_UP = ((-1.5, 0.6), (-0.5, 0.42))
N_LO = ((-1.5, -0.2), (-1.0, -0.4))


def _line(f, p):
    (x1, y1), (x2, y2) = p
    return y1 + (y2 - y1) / (x2 - x1) * (np.asarray(f, float) - x1)


APPLY_LINE = (COL == 'n_fe')
acc_keep = (nfe_all <= _line(feh, N_UP)) & (nfe_all >= _line(feh, N_LO))


def popmask(mk):
    mm = np.asarray(m[mk], bool)
    if mk == 'acc' and APPLY_LINE:
        mm = mm & acc_keep
    return mm


def pctl_tracks(mask):
    edges = np.arange(XR[0], XR[1] + 1e-9, 0.075); cen = 0.5 * (edges[:-1] + edges[1:])
    p5 = np.full(len(cen), np.nan); p95 = np.full(len(cen), np.nan)
    for i in range(len(cen)):
        yy = y[mask & (feh >= edges[i]) & (feh < edges[i+1]) & np.isfinite(y)]
        if yy.size >= 15:
            p5[i], p95[i] = np.percentile(yy, [5, 95])
    return cen, p5, p95


specs = [('acc', 'accreted', c.perc),
         ('thick', r'high-$\alpha$', c.perc2),
         ('thin', r'low-$\alpha$', c.perc2)]

# common log-density stretch across all three top panels (shared colourbar)
hist_cache, mask_cache, dvals = {}, {}, []
for mk, _, perc in specs:
    pop = popmask(mk)
    h, xe, ye = hist2d(feh[pop], y[pop], XR, YR, NFEH, c.nal2, normalize='x')
    hist_cache[mk] = (h, xe, ye)
    mask_cache[mk] = _idl_low_density_mask(h, perc, c.white_lim)
    im = log_image(h); dvals.append(im[np.isfinite(im)])
dvmin, dvmax = finite_percentile(np.concatenate(dvals), (45, 98))

fig, ax = plt.subplots(2, 3, figsize=(12.6, 6.2), sharex=True, sharey=True,
                       constrained_layout=True)

# top row: column-normalised density + P5/P95 (both red)
for j, (mk, title, _) in enumerate(specs):
    h, xe, ye = hist_cache[mk]
    im_top = ax[0, j].imshow(log_image(h).T, origin='lower',
                             extent=[xe[0], xe[-1], ye[0], ye[-1]],
                             aspect='auto', interpolation='nearest',
                             cmap='Greys', vmin=dvmin, vmax=dvmax)
    im_top.set_rasterized(True)
    cen, p5, p95 = pctl_tracks(popmask(mk))
    ax[0, j].plot(cen, p95, color=RED, lw=2.2, zorder=6)
    ax[0, j].plot(cen, p5, color=RED, lw=2.2, zorder=6)
    ax[0, j].set_title(title)

# bottom row: coloured by median V_phi
for j, (mk, title, _) in enumerate(specs):
    h, xe, ye = hist_cache[mk]
    vmask = popmask(mk) & np.isfinite(vt) & (vt >= c.vtanr[0]) & (vt <= c.vtanr[1])
    med, _, _ = stat2d(feh[vmask], y[vmask], vt[vmask], XR, YR, NFEH, c.nal2)
    h_med, _, _ = hist2d(feh[vmask], y[vmask], XR, YR, NFEH, c.nal2)
    med = np.nan_to_num(med, nan=0.0); med[h_med <= 2] = 0.0
    img = np.array(med, float); img[mask_cache[mk]] = np.nan
    im_bot = ax[1, j].imshow(img.T, origin='lower',
                             extent=[xe[0], xe[-1], ye[0], ye[-1]],
                             aspect='auto', interpolation='nearest',
                             cmap='RdYlBu_r', vmin=c.mm_vtan[0], vmax=c.mm_vtan[1])
    im_bot.set_rasterized(True)

for a in ax.ravel():
    a.set_xlim(*XR); a.set_ylim(*YR)
    a.set_aspect('equal')
for a in ax[1, :]:
    a.set_xlabel('[Fe/H]')
for a in ax[:, 0]:
    a.set_ylabel(YLAB)
for a in ax[0, :]:
    a.yaxis.set_major_locator(MaxNLocator(prune='lower'))
for a in ax[1, :]:
    a.yaxis.set_major_locator(MaxNLocator(prune='upper'))

# colourbars at the far right, one per row
cb0 = fig.colorbar(im_top, ax=list(ax[0, :]), location='right', pad=0.02, aspect=22)
cb0.set_label(r'$\log_{10}$ (norm. density)')
cb0.ax.tick_params(length=3)
cb1 = fig.colorbar(im_bot, ax=list(ax[1, :]), location='right', pad=0.02, aspect=22)
cb1.set_label(r'$V_\phi$ [km/s]')
cb1.ax.tick_params(length=3)

# freeze layout, pull the bottom row up to nearly touch the top row, match colourbars
fig.canvas.draw()
fig.set_layout_engine('none')
p_top = ax[0, 0].get_position(); hh = p_top.height; y_top = p_top.y0
GAP = 0.012
y_bot = y_top - hh - GAP
for a in ax[1, :]:
    pa = a.get_position(); a.set_position([pa.x0, y_bot, pa.width, hh])
for cb, y0 in ((cb0, y_top), (cb1, y_bot)):
    cp = cb.ax.get_position(); cb.ax.set_position([cp.x0, y0, cp.width, hh])

for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/{OUTNAME}.{ext}', bbox_inches='tight')
print('wrote', OUT + f'/{OUTNAME}.{{pdf,png}}')
for mk, title, _ in specs:
    print(f'  {title:10s} n={int(popmask(mk).sum())}')
