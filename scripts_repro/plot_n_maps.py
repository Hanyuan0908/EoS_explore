"""MAPs-style panel: decompose the [Fe/H]-[Mg/Fe] plane into pixels and draw the
[N/Fe] distribution inside each pixel (x-within-cell = [N/Fe], y = density), coloured
by the cell's median [N/Fe]. Grey background = log density of the base sample. A faint
vertical line marks [N/Fe]=0 in every cell so the shift / high-N wing is visible.
Standard global-quality sample (no per-element N_FE_FLAG cut).
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm, colors
from scipy.stats import gaussian_kde
REPO = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/eos-figures')
sys.path.insert(0, str(REPO))
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
c = Cuts()
FIG = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/figures_repro')
cat = load_catalog('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c)
base = np.asarray(m['base'], bool)
feh = np.asarray(cat['fe_h'], float); mg = np.asarray(cat['mg_fe'], float)
nfe = np.asarray(cat['n_fe'], float); vphi = np.asarray(cat['galvt'], float)

FEHR = (-1.25, 0.55); MGR = (-0.05, 0.45)
DFEH, DMG = 0.10, 0.05
NLO, NHI = -0.35, 0.85          # [N/Fe] axis inside each cell
cmap = cm.viridis; norm = colors.Normalize(-0.10, 0.45)


def make_map(good, nmin, title, fname, annotate=False):
    from matplotlib.patches import Ellipse
    fig, ax = plt.subplots(figsize=(14, 9.5), constrained_layout=True)
    h, xe, ye = np.histogram2d(feh[good], mg[good], bins=[90, 60], range=[FEHR, MGR])
    himg = np.full_like(h, np.nan); himg[h > 0] = np.log10(h[h > 0])  # Davies-style log-density
    ax.imshow(himg.T, origin='lower', extent=[*FEHR, *MGR], aspect='auto', cmap='Greys',
              alpha=0.72, vmin=np.nanpercentile(himg, 3), vmax=np.nanpercentile(himg, 99.5), zorder=0)
    if annotate:
        xx = np.array(FEHR)
        ax.plot(xx, c.slope_acc * xx + c.inter_acc, 'r--', lw=1.3, zorder=4)      # accreted / in-situ
        ax.plot(xx, c.slope_acc2 * xx + c.inter_acc2, 'r:', lw=1.6, zorder=4)      # high / low-alpha
        ax.add_patch(Ellipse((-0.68, 0.155), 0.44, 0.13, angle=-8, fill=False, ec='red', lw=2.6, zorder=5))
        for tx, ty, lab, col in [(-0.66, 0.235, 'Eos', 'red'),
                                 (-0.42, 0.31, 'Splash (heated high-$\\alpha$)', 'k'),
                                 (-1.18, 0.335, 'Aurora', 'k'),
                                 (-1.20, 0.06, 'GS/E (accreted)', 'k'),
                                 (0.15, 0.015, 'heated low-$\\alpha$ disc', 'k')]:
            ax.text(tx, ty, lab, color=col, fontsize=10, fontweight='bold', zorder=6)
    fedges = np.arange(FEHR[0], FEHR[1] + 1e-9, DFEH); medges = np.arange(MGR[0], MGR[1] + 1e-9, DMG)
    xn = np.linspace(NLO, NHI, 60); ncell = 0
    for fx0 in fedges[:-1]:
        for my0 in medges[:-1]:
            sel = good & (feh >= fx0) & (feh < fx0 + DFEH) & (mg >= my0) & (mg < my0 + DMG)
            if sel.sum() < nmin:
                continue
            y = nfe[sel]
            d = gaussian_kde(y)(xn); d = d / d.max()
            cx = fx0 + (xn - NLO) / (NHI - NLO) * DFEH
            cy = my0 + 0.04 * DMG + d * 0.88 * DMG
            x0 = fx0 + (0 - NLO) / (NHI - NLO) * DFEH
            ax.plot([x0, x0], [my0, my0 + DMG], color='0.55', lw=0.3, alpha=0.6, zorder=1)
            ax.add_patch(Rectangle((fx0, my0), DFEH, DMG, fill=False, ec='0.6', lw=0.3, zorder=1))
            ax.plot(cx, cy, color=cmap(norm(np.median(y))), lw=1.1, zorder=2)
            ncell += 1
    ax.set_xlim(*FEHR); ax.set_ylim(*MGR)
    ax.set_xlabel('[Fe/H] [dex]'); ax.set_ylabel('[Mg/Fe] [dex]')
    ax.set_title(f'{title} ({ncell} cells, N$\\geq${nmin};  x-in-cell = [N/Fe] over [{NLO},{NHI}], grey line = [N/Fe]=0)')
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.01).set_label('median [N/Fe] in cell [dex]')
    fig.savefig(FIG / fname, dpi=160, bbox_inches='tight'); plt.close(fig)
    print(f'drew {ncell} cells; wrote', FIG / fname)


fin = np.isfinite(feh) & np.isfinite(mg) & np.isfinite(nfe)
make_map(base & fin, 25, 'MAPs: [N/Fe] distribution per [Fe/H]-[Mg/Fe] pixel (all base sample)', '01_n_maps.png')
make_map(base & fin & np.isfinite(vphi) & (vphi < 80), 15,
         r'MAPs: [N/Fe] per [Fe/H]-[Mg/Fe] pixel -- HALO-like ($V_{\rm tan}<80$)', '01_n_maps_halo.png',
         annotate=True)
