"""Reference Fig-2 layout for [N/Fe] (accreted / high-a / low-a): top row = column-normalised
[N/Fe]-[Fe/H] density, bottom row = coloured by median V_tan. Remake with:
  - [Fe/H] lower limit changed to -1.5
  - the 5th and 95th percentile of [N/Fe] drawn on each TOP panel (vs [Fe/H]).
Standalone reproduction of build_nb.py::pops_plane.
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
from eos_figures.plotting import setup_axes, density_panel, value_panel, label_axes
from eos_figures.figures import _idl_low_density_mask
c = Cuts()
FIG = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/figures_repro')
cat = load_catalog('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c)

# element parametrised: `python plot_fig2_nfe_pops.py [n_fe|c_fe]`
COL = sys.argv[1] if len(sys.argv) > 1 else 'n_fe'
PARAMS = {'n_fe': ((-0.5, 1.0), '[N/Fe]', '01_fig2_nfe_pops.png'),
          'c_fe': ((-0.6, 0.4), '[C/Fe]', '01_fig2_cfe_pops.png')}
YR, YLAB, OUTNAME = PARAMS[COL]
FEHR = (-1.5, c.fehr[1])                 # lower limit changed to -1.5
NFEH = int(round((FEHR[1] - FEHR[0]) / ((c.fehr[1] - c.fehr[0]) / c.nfeh)))   # keep bin width
y = np.asarray(cat[COL], float); feh = np.asarray(cat['fe_h'], float); vt = np.asarray(cat['galvt'], float)

# P5/P95 (and median) of [N/Fe] vs [Fe/H], per population, from the raw stars
def pctl_tracks(mask):
    edges = np.arange(FEHR[0], FEHR[1] + 1e-9, 0.075); cen = 0.5 * (edges[:-1] + edges[1:])
    p5 = np.full(len(cen), np.nan); p50 = np.full(len(cen), np.nan); p95 = np.full(len(cen), np.nan)
    for i in range(len(cen)):
        yy = y[mask & (feh >= edges[i]) & (feh < edges[i+1]) & np.isfinite(y)]
        if yy.size >= 15:
            p5[i], p50[i], p95[i] = np.percentile(yy, [5, 50, 95])
    return cen, p5, p50, p95

specs = [('acc', 'accreted', c.perc), ('thick', r'high-$\alpha$', c.perc2), ('thin', r'low-$\alpha$', c.perc2)]
fig, ax = setup_axes(3, nrows=2, figsize=(10, 6))
hist_cache, mask_cache = {}, {}
for i, (mk, title, perc) in enumerate(specs):
    h, xe, ye = hist2d(feh[m[mk]], y[m[mk]], FEHR, YR, NFEH, c.nal2, normalize='x')
    hist_cache[mk] = (h, xe, ye); mask_cache[mk] = _idl_low_density_mask(h, perc, c.white_lim)
    density_panel(ax[i], h, xe, ye, percentiles=perc)
    # --- the two requested lines: P5 and P95 of [N/Fe] (+ faint median) ---
    cen, p5, p50, p95 = pctl_tracks(np.asarray(m[mk], bool))
    ax[i].plot(cen, p95, color='crimson', lw=2.2, zorder=6, label='P95')
    ax[i].plot(cen, p5, color='royalblue', lw=2.2, zorder=6, label='P5')
    ax[i].plot(cen, p50, color='0.4', lw=1.2, ls='--', zorder=6, label='median')
    ax[i].set_xlim(*FEHR); ax[i].set_ylim(*YR)
    label_axes(ax[i], '[Fe/H]', YLAB, title)
    if i == 2:
        ax[i].legend(frameon=False, fontsize=8, loc='upper right')
# bottom row: coloured by median V_tan (unchanged)
for j, (mk, title, _) in enumerate(specs):
    i = j + 3
    h, xe, ye = hist_cache[mk]
    vmask = np.asarray(m[mk], bool) & np.isfinite(vt) & (vt >= c.vtanr[0]) & (vt <= c.vtanr[1])
    med, _, _ = stat2d(feh[vmask], y[vmask], vt[vmask], FEHR, YR, NFEH, c.nal2)
    h_med, _, _ = hist2d(feh[vmask], y[vmask], FEHR, YR, NFEH, c.nal2)
    med = np.nan_to_num(med, nan=0.0); med[h_med <= 2] = 0.0
    value_panel(ax[i], med, xe, ye, *c.mm_vtan, mask=mask_cache[mk],
                cmap='RdYlBu_r', colorbar_label=r'$V_{\rm tan}$ [km/s]' if i == 4 else None)
    ax[i].set_xlim(*FEHR); ax[i].set_ylim(*YR)
    label_axes(ax[i], '[Fe/H]', YLAB, title)
ax[5].text(-1.3, YR[0] + 0.85 * (YR[1] - YR[0]), 'Eos?', fontsize=9)
fig.savefig(FIG / OUTNAME, dpi=150, bbox_inches='tight')
print('wrote', FIG / OUTNAME)
for mk, title, _ in specs:
    cen, p5, p50, p95 = pctl_tracks(np.asarray(m[mk], bool))
    print(f'{title:10s} P95-P5 at [Fe/H]=-1.2,-0.8,-0.4:',
          [round(np.interp(v, cen, p95) - np.interp(v, cen, p5), 2) for v in (-1.2, -0.8, -0.4)])
