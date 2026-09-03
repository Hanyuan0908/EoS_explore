"""Same Davies halo chemistry planes, coloured by MEAN orbital eccentricity
e = (r_apo - r_peri)/(r_apo + r_peri) (AstroNN), per bin. Bounded [0,1], well-behaved,
so the mean is a fine summary (no heavy tail / no AGAMA needed).
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
REPO = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/eos-figures')
sys.path.insert(0, str(REPO))
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
from eos_figures.plotting import label_axes
c = Cuts()
FIG = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/figures_repro')
cat = load_catalog('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c); base = np.asarray(m['base'], bool)
feh = np.asarray(cat['fe_h'], float); mg = np.asarray(cat['mg_fe'], float); al = np.asarray(cat['al_fe'], float)
lz = np.asarray(cat['lz'], float); rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float)
ecc = (rap - rperi)/(rap + rperi)
halo = base & ((ecc > 0.7) | (lz < 0))
sel = halo & np.isfinite(ecc)
print('halo n=', int(sel.sum()), ' mean ecc=', np.round(np.mean(ecc[sel]), 3))

STAT = sys.argv[1] if len(sys.argv) > 1 else 'mean'      # 'mean' or 'median'
XR = (-2.1, 0.6); NMIN = 3
VMIN, VMAX = 0.8, 0.95                     # eccentricity colour range (data sits in a narrow high-e band)
def panel(ax, yv, yr, ylab, nb=(45, 35)):
    s = sel & np.isfinite(feh) & np.isfinite(yv)
    me = binned_statistic_2d(feh[s], yv[s], ecc[s], statistic=STAT, bins=nb, range=[XR, yr]).statistic
    cnt = binned_statistic_2d(feh[s], yv[s], None, statistic='count', bins=nb, range=[XR, yr]).statistic
    mj = np.where(cnt >= NMIN, me, np.nan)
    im = ax.imshow(mj.T, origin='lower', extent=[*XR, *yr], aspect='auto', cmap='coolwarm',
                   vmin=VMIN, vmax=VMAX, zorder=0)
    ax.set_xlim(*XR); ax.set_ylim(*yr)
    label_axes(ax, '[Fe/H]', ylab, f'Davies halo, coloured by {STAT} eccentricity (n={int(s.sum())})')
    return im

def mg_lines(ax):
    xx = np.linspace(-2.1, 0.6, 50)
    ax.plot(xx, c.slope_acc*xx + c.inter_acc, color='k', ls='--', lw=1.6, zorder=3, label='accreted / in-situ')
    ax.plot(xx, c.slope_acc2*xx + c.inter_acc2, color='k', ls=':', lw=2.0, zorder=3, label=r'high-$\alpha$ / low-$\alpha$')
    xe = np.linspace(-0.9, -0.2, 30)                                  # Davies divider: splits the two Eos branches
    ax.plot(xe, 0.317*xe + 0.353, color='lime', ls='-', lw=2.0, zorder=4, label=r'Eos divider ($\alpha$-rich/$\alpha$-poor)')
    leg = ax.legend(frameon=False, fontsize=8, loc='lower left')
    for t in leg.get_texts(): t.set_color('black')

fig, ax = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
panel(ax[0], al, (-0.45, 0.45), '[Al/Fe]')
im = panel(ax[1], mg, (-0.1, 0.5), '[Mg/Fe]')
mg_lines(ax[1])
ax[1].text(-0.62, 0.16, 'Eos', color='k', fontsize=13, fontweight='bold', zorder=3)
cb = fig.colorbar(im, ax=ax, pad=0.01, shrink=0.9); cb.set_label(f'{STAT} eccentricity in bin')
fig.suptitle(f'{STAT.capitalize()} orbital eccentricity across the halo chemistry planes (AstroNN)', fontsize=12)
out = '01_davies_fig2_ecc.png' if STAT == 'mean' else f'01_davies_fig2_ecc_{STAT}.png'
fig.savefig(FIG / out, dpi=150, bbox_inches='tight')
print('wrote', FIG / out)
