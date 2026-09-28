"""Non-publish sweep: reproduce the obs_nfe_pops figure (Fig_code/obs_nfe_pops)
in the EXACT same style, for every usable APOGEE [X/Fe] species.

Per species, a 2x2 panel (columns = high-alpha / low-alpha):
  top row    : column-normalised [X/Fe]-[Fe/H] density (Greys) with the 5th and
               95th percentile tracks of [X/Fe] vs [Fe/H] (both red).
  bottom row : the same planes coloured by median V_phi (RdYlBu_r).
Shared log-density stretch across the two top panels; colourbars at the far right,
one per row; equal physical aspect; near-zero row gap. Same masks (`thin`/`thick`),
same helpers, same rcParams as the publication figure.

Only per-species change: the [X/Fe] y-range is auto-framed from the in-situ data
(padded 0.5-99.5 percentiles, min span 1.0 dex) so equal aspect stays readable.

Data: data_repro/our_apogee_allspecies.fits.gz (19-species wide cache, Mac-only).
Writes one PNG per species into figures_repro/species_pops/<col>.png.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
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

DATA = REPO + '/data_repro/our_apogee_allspecies.fits.gz'
OUTD = Path(REPO + '/figures_repro/species_pops')
OUTD.mkdir(parents=True, exist_ok=True)
RED = '#E8112D'

SPECIES = ['c_fe', 'ci_fe', 'n_fe', 'o_fe', 'na_fe', 'mg_fe', 'al_fe', 'si_fe', 's_fe',
           'k_fe', 'ca_fe', 'ti_fe', 'tiii_fe', 'v_fe', 'cr_fe', 'mn_fe', 'co_fe', 'ni_fe', 'ce_fe']
PRETTY = {'c_fe': '[C/Fe]', 'ci_fe': '[C I/Fe]', 'n_fe': '[N/Fe]', 'o_fe': '[O/Fe]',
          'na_fe': '[Na/Fe]', 'mg_fe': '[Mg/Fe]', 'al_fe': '[Al/Fe]', 'si_fe': '[Si/Fe]',
          's_fe': '[S/Fe]', 'k_fe': '[K/Fe]', 'ca_fe': '[Ca/Fe]', 'ti_fe': '[Ti/Fe]',
          'tiii_fe': '[Ti II/Fe]', 'v_fe': '[V/Fe]', 'cr_fe': '[Cr/Fe]', 'mn_fe': '[Mn/Fe]',
          'co_fe': '[Co/Fe]', 'ni_fe': '[Ni/Fe]', 'ce_fe': '[Ce/Fe]'}

c = Cuts()
cat = load_catalog(DATA)
m = make_masks(cat, c)
feh = np.asarray(cat['fe_h'], float)
vt = np.asarray(cat['galvt'], float)
XR = (-1.5, c.fehr[1])                                             # [Fe/H] range (as obs_nfe_pops)
NFEH = int(round((XR[1] - XR[0]) / ((c.fehr[1] - c.fehr[0]) / c.nfeh)))  # keep bin width
insitu = np.asarray(m['thin'], bool) | np.asarray(m['thick'], bool)
specs = [('thick', r'high-$\alpha$'), ('thin', r'low-$\alpha$')]


def auto_yrange(y):
    yy = y[insitu & np.isfinite(y) & (feh >= XR[0]) & (feh < XR[1])]
    if yy.size < 50:
        return (-0.5, 0.7)
    lo, hi = np.percentile(yy, [0.5, 99.5])
    lo -= 0.08; hi += 0.08
    if hi - lo < 1.0:                                              # floor span for equal-aspect readability
        mid = 0.5 * (lo + hi); lo, hi = mid - 0.5, mid + 0.5
    return (np.floor(lo / 0.05) * 0.05, np.ceil(hi / 0.05) * 0.05)


def pctl_tracks(mask, y, yr):
    edges = np.arange(XR[0], XR[1] + 1e-9, 0.075); cen = 0.5 * (edges[:-1] + edges[1:])
    p5 = np.full(len(cen), np.nan); p95 = np.full(len(cen), np.nan)
    for i in range(len(cen)):
        yy = y[mask & (feh >= edges[i]) & (feh < edges[i + 1]) & np.isfinite(y)]
        if yy.size >= 15:
            p5[i], p95[i] = np.percentile(yy, [5, 95])
    return cen, p5, p95


def make_one(col):
    y = np.asarray(cat[col], float)
    YR = auto_yrange(y)
    YLAB = PRETTY.get(col, '[' + col + ']')

    # shared log-density stretch across both top panels
    hist_cache, mask_cache, dvals = {}, {}, []
    for mk, _ in specs:
        h, xe, ye = hist2d(feh[m[mk]], y[m[mk]], XR, YR, NFEH, c.nal2, normalize='x')
        hist_cache[mk] = (h, xe, ye)
        mask_cache[mk] = _idl_low_density_mask(h, c.perc2, c.white_lim)
        im = log_image(h); dvals.append(im[np.isfinite(im)])
    dvmin, dvmax = finite_percentile(np.concatenate(dvals), c.perc2)

    fig, ax = plt.subplots(2, 2, figsize=(8.8, 6.4), sharex=True, sharey=True,
                           constrained_layout=True)

    im_top = im_bot = None
    for j, (mk, title) in enumerate(specs):
        h, xe, ye = hist_cache[mk]
        im_top = ax[0, j].imshow(log_image(h).T, origin='lower',
                                 extent=[xe[0], xe[-1], ye[0], ye[-1]],
                                 aspect='auto', interpolation='nearest',
                                 cmap='Greys', vmin=dvmin, vmax=dvmax)
        im_top.set_rasterized(True)
        cen, p5, p95 = pctl_tracks(np.asarray(m[mk], bool), y, YR)
        ax[0, j].plot(cen, p95, color=RED, lw=2.2, zorder=6)
        ax[0, j].plot(cen, p5, color=RED, lw=2.2, zorder=6)
        ax[0, j].set_title(title)

    for j, (mk, title) in enumerate(specs):
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
        a.set_xlim(*XR); a.set_ylim(*YR); a.set_aspect('equal')
    for a in ax[1, :]:
        a.set_xlabel('[Fe/H]')
    for a in ax[:, 0]:
        a.set_ylabel(YLAB)
    for a in ax[0, :]:
        a.yaxis.set_major_locator(MaxNLocator(prune='lower'))
    for a in ax[1, :]:
        a.yaxis.set_major_locator(MaxNLocator(prune='upper'))

    cb0 = fig.colorbar(im_top, ax=list(ax[0, :]), location='right', pad=0.02, aspect=22)
    cb0.set_label(r'$\log_{10}$ (norm. density)'); cb0.ax.tick_params(length=3)
    cb1 = fig.colorbar(im_bot, ax=list(ax[1, :]), location='right', pad=0.02, aspect=22)
    cb1.set_label(r'$V_\phi$ [km/s]'); cb1.ax.tick_params(length=3)

    fig.canvas.draw(); fig.set_layout_engine('none')
    p_top = ax[0, 0].get_position(); hh = p_top.height; y_top = p_top.y0
    GAP = 0.012; y_bot = y_top - hh - GAP
    for a in ax[1, :]:
        pa = a.get_position(); a.set_position([pa.x0, y_bot, pa.width, hh])
    for cb, y0 in ((cb0, y_top), (cb1, y_bot)):
        cp = cb.ax.get_position(); cb.ax.set_position([cp.x0, y0, cp.width, hh])

    fig.savefig(OUTD / f'{col}.png', bbox_inches='tight')
    plt.close(fig)
    return YR


if __name__ == '__main__':
    todo = sys.argv[1:] or SPECIES
    for col in todo:
        if col not in cat.dtype.names:
            print(f'skip {col}: not in cache'); continue
        yr = make_one(col)
        print(f'wrote {OUTD}/{col}.png  (y-range {yr[0]:+.2f},{yr[1]:+.2f})')
