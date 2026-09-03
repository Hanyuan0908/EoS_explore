"""Publication figure (observational): [N/Fe]-[Fe/H] for the two in-situ
populations (high-alpha / low-alpha), 2x2.

  top row    : column-normalised [N/Fe]-[Fe/H] density (Greys), with the
               5th and 95th percentile tracks of [N/Fe] vs [Fe/H] (both red).
  bottom row : the same planes coloured by median V_phi.

Colourbars sit at the far right, one per row (log density on top, V_phi below).
Panels use equal physical aspect (1 dex in [Fe/H] == 1 dex in [N/Fe]).

Derived from scripts_repro/plot_fig2_nfe_pops.py; accreted column dropped,
legend / median line / "Eos?" annotation removed per request.

Data: data_repro/our_apogee_dr17_lite_ann.fits.gz (in-repo, portable).
`python fig_paper_nfe_pops.py [n_fe|c_fe]`  (default n_fe)
Writes Fig_paper/obs_nfe_pops.pdf and .png.
"""
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

REPO = '/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore'
sys.path.insert(0, REPO + '/eos-figures')
# imports first, then our own rcParams below so the publication style wins over
# the reference repo's (applied when eos_figures.figures imports plotting).
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

# element parametrised: `python fig_paper_nfe_pops.py [n_fe|c_fe]`
COL = sys.argv[1] if len(sys.argv) > 1 else 'n_fe'
PARAMS = {'n_fe': ((-0.5, 0.7), '[N/Fe]', 'obs_nfe_pops'),
          'c_fe': ((-0.6, 0.4), '[C/Fe]', 'obs_cfe_pops')}
YR, YLAB, OUTNAME = PARAMS[COL]
XR = (-1.5, c.fehr[1])                                          # [Fe/H] range
NFEH = int(round((XR[1] - XR[0]) / ((c.fehr[1] - c.fehr[0]) / c.nfeh)))  # keep bin width
RED = '#E8112D'

y = np.asarray(cat[COL], float)
feh = np.asarray(cat['fe_h'], float)
vt = np.asarray(cat['galvt'], float)


def pctl_tracks(mask):
    edges = np.arange(XR[0], XR[1] + 1e-9, 0.075); cen = 0.5 * (edges[:-1] + edges[1:])
    p5 = np.full(len(cen), np.nan); p95 = np.full(len(cen), np.nan)
    for i in range(len(cen)):
        yy = y[mask & (feh >= edges[i]) & (feh < edges[i+1]) & np.isfinite(y)]
        if yy.size >= 15:
            p5[i], p95[i] = np.percentile(yy, [5, 95])
    return cen, p5, p95


specs = [('thick', r'high-$\alpha$', c.perc2), ('thin', r'low-$\alpha$', c.perc2)]

# common log-density stretch across both top panels (shared colourbar)
hist_cache, mask_cache, dvals = {}, {}, []
for mk, _, perc in specs:
    h, xe, ye = hist2d(feh[m[mk]], y[m[mk]], XR, YR, NFEH, c.nal2, normalize='x')
    hist_cache[mk] = (h, xe, ye)
    mask_cache[mk] = _idl_low_density_mask(h, perc, c.white_lim)
    im = log_image(h); dvals.append(im[np.isfinite(im)])
dvmin, dvmax = finite_percentile(np.concatenate(dvals), specs[0][2])

fig, ax = plt.subplots(2, 2, figsize=(8.8, 6.4), sharex=True, sharey=True,
                       constrained_layout=True)

# top row: column-normalised density + P5/P95 (both red)
for j, (mk, title, _) in enumerate(specs):
    h, xe, ye = hist_cache[mk]
    im_top = ax[0, j].imshow(log_image(h).T, origin='lower',
                             extent=[xe[0], xe[-1], ye[0], ye[-1]],
                             aspect='auto', interpolation='nearest',
                             cmap='Greys', vmin=dvmin, vmax=dvmax)
    im_top.set_rasterized(True)
    cen, p5, p95 = pctl_tracks(np.asarray(m[mk], bool))
    ax[0, j].plot(cen, p95, color=RED, lw=2.2, zorder=6)
    ax[0, j].plot(cen, p5, color=RED, lw=2.2, zorder=6)
    ax[0, j].set_title(title)

# bottom row: coloured by median V_phi
for j, (mk, title, _) in enumerate(specs):
    h, xe, ye = hist_cache[mk]
    vmask = np.asarray(m[mk], bool) & np.isfinite(vt) & (vt >= c.vtanr[0]) & (vt <= c.vtanr[1])
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
    a.set_aspect('equal')                # 1 dex [Fe/H] == 1 dex [N/Fe]
for a in ax[1, :]:
    a.set_xlabel('[Fe/H]')
for a in ax[:, 0]:
    a.set_ylabel(YLAB)
# near-zero row gap: prune the boundary y-tick label so the two rows don't collide
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

# with aspect='equal' the panel boxes shrink vertically inside their slots, leaving
# a large row gap and colourbars taller than the panels. Freeze the layout, pull
# the bottom row up to nearly touch the top row, and match each colourbar to its
# row's panel box.
fig.canvas.draw()
fig.set_layout_engine('none')
p_top = ax[0, 0].get_position(); h = p_top.height; y_top = p_top.y0
GAP = 0.012                                    # near-zero row gap (figure fraction)
y_bot = y_top - h - GAP
for a in ax[1, :]:                             # move bottom row up
    pa = a.get_position(); a.set_position([pa.x0, y_bot, pa.width, h])
for cb, y0 in ((cb0, y_top), (cb1, y_bot)):    # colourbars follow their rows
    cp = cb.ax.get_position(); cb.ax.set_position([cp.x0, y0, cp.width, h])

for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/{OUTNAME}.{ext}', bbox_inches='tight')
print('wrote', OUT + f'/{OUTNAME}.{{pdf,png}}')
