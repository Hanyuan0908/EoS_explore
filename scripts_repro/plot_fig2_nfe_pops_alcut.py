"""Non-publish: the Fig-2 [N/Fe] population planes (accreted / high-a / low-a),
produced in TWO variants for comparison:
  - 01_fig2_nfe_pops_noAlcut.png : high-a/low-a via `thick`/`thin`   (no [Al/Fe] floor)
  - 01_fig2_nfe_pops_Alcut.png   : high-a/low-a via `thick_al`/`thin_al`
                                   (in-situ [Al/Fe] > -0.12 applied)
The accreted column uses `acc` in BOTH (the [Al/Fe]>-0.12 floor is an in-situ
selector and does not apply to the accreted population, which is [Al/Fe]<-0.12).
Everything else (layout, P5/P95/median tracks, V_tan bottom row) is identical to
scripts_repro/plot_fig2_nfe_pops.py.

`python plot_fig2_nfe_pops_alcut.py [n_fe|c_fe]`  (default n_fe).
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
REPO = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/eos-figures')
sys.path.insert(0, str(REPO))
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
from eos_figures.stats import hist2d, stat2d
from eos_figures.plotting import setup_axes, density_panel, value_panel, label_axes, log_image
from eos_figures.stats import finite_percentile
from eos_figures.figures import _idl_low_density_mask
c = Cuts()
FIG = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/figures_repro')
cat = load_catalog('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c)

COL = sys.argv[1] if len(sys.argv) > 1 else 'n_fe'
PARAMS = {'n_fe': ((-0.5, 1.0), '[N/Fe]', 'nfe'),
          'c_fe': ((-0.6, 0.4), '[C/Fe]', 'cfe')}
YR, YLAB, TAG = PARAMS[COL]
FEHR = (-1.5, c.fehr[1])
NFEH = int(round((FEHR[1] - FEHR[0]) / ((c.fehr[1] - c.fehr[0]) / c.nfeh)))
y = np.asarray(cat[COL], float); feh = np.asarray(cat['fe_h'], float); vt = np.asarray(cat['galvt'], float)

# --- accreted chemical outlier cut (N/Fe figure only) ------------------------------
# The `acc` mask already applies |Lz| < 500 (lz_lim_acc) and every accreted star has a
# finite V_phi, so the high-/low-[N/Fe] stragglers are CHEMICAL outliers, not kinematic
# ones. Remove accreted stars above the line (-1.5, 0.6)->(-0.5, 0.42) or below the line
# (-1.5, -0.2)->(-1.0, -0.4) in the [N/Fe]-[Fe/H] plane. Applied to the accreted panel
# only (discs untouched) and only for the [N/Fe] figure; the lines are drawn on the panel.
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


# two variants: only the in-situ (high/low-a) masks change; accreted stays `acc`
VARIANTS = [
    ('noAlcut', [('acc', 'accreted'), ('thick', r'high-$\alpha$'), ('thin', r'low-$\alpha$')],
     r'no [Al/Fe] cut'),
    ('Alcut',   [('acc', 'accreted'), ('thick_al', r'high-$\alpha$'), ('thin_al', r'low-$\alpha$')],
     r'in-situ [Al/Fe] $> -0.12$ applied'),
]
PERC = {'acc': c.perc, 'thick': c.perc2, 'thin': c.perc2, 'thick_al': c.perc2, 'thin_al': c.perc2}


def pctl_tracks(mask):
    edges = np.arange(FEHR[0], FEHR[1] + 1e-9, 0.075); cen = 0.5 * (edges[:-1] + edges[1:])
    p5 = np.full(len(cen), np.nan); p50 = np.full(len(cen), np.nan); p95 = np.full(len(cen), np.nan)
    for i in range(len(cen)):
        yy = y[mask & (feh >= edges[i]) & (feh < edges[i + 1]) & np.isfinite(y)]
        if yy.size >= 15:
            p5[i], p50[i], p95[i] = np.percentile(yy, [5, 50, 95])
    return cen, p5, p50, p95


for suffix, specs, note in VARIANTS:
    fig, ax = setup_axes(3, nrows=2, figsize=(10.6, 6))
    hist_cache, mask_cache = {}, {}
    # First pass: build all three column-normalised histograms, then derive ONE shared
    # colour normalisation (vmin/vmax on the log column-normalised density) from the
    # pooled pixel values, so all three top panels use the SAME greyscale -- the
    # accreted panel no longer gets a darker clip than the two disc panels.
    logims = {}
    for mk, title in specs:
        pop = popmask(mk)
        h, xe, ye = hist2d(feh[pop], y[pop], FEHR, YR, NFEH, c.nal2, normalize='x')
        hist_cache[mk] = (h, xe, ye)
        mask_cache[mk] = _idl_low_density_mask(h, PERC[mk], c.white_lim)
        li = log_image(h); logims[mk] = li
    allpix = np.concatenate([li[np.isfinite(li)].ravel() for li in logims.values()])
    vlo, vhi = finite_percentile(allpix, (45, 98))
    # top row: column-normalised density, shared greyscale
    im_d = None
    for i, (mk, title) in enumerate(specs):
        h, xe, ye = hist_cache[mk]
        im_d = density_panel(ax[i], h, xe, ye, vmin=vlo, vmax=vhi)
        cen, p5, p50, p95 = pctl_tracks(popmask(mk))
        ax[i].plot(cen, p95, color='crimson', lw=2.2, zorder=6, label='P95')
        ax[i].plot(cen, p5, color='royalblue', lw=2.2, zorder=6, label='P5')
        ax[i].plot(cen, p50, color='0.4', lw=1.2, ls='--', zorder=6, label='median')
        if mk == 'acc' and APPLY_LINE:  # draw the two outlier-rejection lines
            fu = np.array([-1.5, -0.2]); ax[i].plot(fu, _line(fu, N_UP), color='#2E7D32', ls='--', lw=1.4, zorder=7)
            fl = np.array([-1.5, -0.75]); ax[i].plot(fl, _line(fl, N_LO), color='#2E7D32', ls='--', lw=1.4, zorder=7)
        ax[i].set_xlim(*FEHR); ax[i].set_ylim(*YR)
        label_axes(ax[i], '[Fe/H]', YLAB, title)
        if i == 2:
            ax[i].legend(frameon=False, fontsize=8, loc='upper right')
    cb_d = fig.colorbar(im_d, ax=[ax[0], ax[1], ax[2]], location='right', pad=0.012, aspect=30)
    cb_d.set_label('log column-norm. density', fontsize=9); cb_d.ax.tick_params(labelsize=8)
    # bottom row: median V_tan, single shared colourbar (same vmin/vmax already)
    im_v = None
    for j, (mk, title) in enumerate(specs):
        i = j + 3
        h, xe, ye = hist_cache[mk]
        vmask = popmask(mk) & np.isfinite(vt) & (vt >= c.vtanr[0]) & (vt <= c.vtanr[1])
        med, _, _ = stat2d(feh[vmask], y[vmask], vt[vmask], FEHR, YR, NFEH, c.nal2)
        h_med, _, _ = hist2d(feh[vmask], y[vmask], FEHR, YR, NFEH, c.nal2)
        med = np.nan_to_num(med, nan=0.0); med[h_med <= 2] = 0.0
        im_v = value_panel(ax[i], med, xe, ye, *c.mm_vtan, mask=mask_cache[mk], cmap='RdYlBu_r')
        ax[i].set_xlim(*FEHR); ax[i].set_ylim(*YR)
        label_axes(ax[i], '[Fe/H]', YLAB, title)
    cb_v = fig.colorbar(im_v, ax=[ax[3], ax[4], ax[5]], location='right', pad=0.012, aspect=30)
    cb_v.set_label(r'$V_{\rm tan}$ [km/s]', fontsize=9); cb_v.ax.tick_params(labelsize=8)
    ax[5].text(-1.3, YR[0] + 0.85 * (YR[1] - YR[0]), 'Eos?', fontsize=9)
    fig.text(0.01, 1.02, note, fontsize=12, ha='left', va='bottom', fontweight='bold')
    _note2 = 'shared greyscale' + ('; accreted N/Fe outlier lines (green)' if APPLY_LINE else '')
    fig.text(0.99, 1.02, _note2, fontsize=8.5, ha='right', va='bottom', color='0.3')
    out = FIG / f'01_fig2_{TAG}_pops_{suffix}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('wrote', out)
    for mk, title in specs:
        print(f'  [{suffix}] {title:10s} n={int(popmask(mk).sum())}')
