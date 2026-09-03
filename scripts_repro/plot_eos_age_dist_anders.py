"""Eos age distribution with ANDERS+2023 spectroscopic ages (spAgeqrCal), matched by APOGEE_ID.
Same construction as plot_eos_age_dist.py (AstroNN). Canonical Eos cut, two branches, disc by [Fe/H].
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy.stats import gaussian_kde
from astropy.io import fits
REPO = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/eos-figures')
sys.path.insert(0, str(REPO))
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
from eos_figures.plotting import label_axes
c = Cuts()
FIG = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/figures_repro')
cat = load_catalog('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c)
feh = np.asarray(cat['fe_h'], float); mg = np.asarray(cat['mg_fe'], float); vphi = np.asarray(cat['galvt'], float)
al = np.asarray(cat['al_fe'], float); lz = np.asarray(cat['lz'], float)
rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float); ecc = (rap - rperi)/(rap + rperi)
aid = np.asarray(cat['apogee_id']); base = np.asarray(m['base'], bool); thin_al = np.asarray(m['thin_al'], bool)
halo = base & ((ecc > 0.7) | (lz < 0))

# --- ANDERS 2023 ages by APOGEE_ID ---
d = fits.open('/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE/APOGEE_AstroNNdist_Anders23age_BJdist.fits')[1].data
def norm(a): return np.array([(s.decode() if isinstance(s, bytes) else str(s)).strip() for s in np.asarray(a)])
did = norm(d['APOGEE_ID_2']); o = np.argsort(did); dids = did[o]
p = np.clip(np.searchsorted(dids, aid), 0, len(dids)-1); ok = dids[p] == aid; src = o[p]
age = np.where(ok, np.asarray(d['spAgeqrCal'], float)[src], np.nan)
rel_ok = np.isfinite(age) & (age > 0) & (age < 20)

def acc(f): return c.slope_acc*f + c.inter_acc
def hl(f): return c.slope_acc2*f + c.inter_acc2
def divline(f): return 0.317*f + 0.353
eos = halo & (feh > -0.9) & (feh < -0.2) & (mg > acc(feh)) & (mg < hl(feh)) & (al > c.alfe_cut)
eos_hi = eos & (mg > divline(feh)); eos_lo = eos & (mg <= divline(feh))
disc = thin_al & (vphi > 150)
CHI, CLO = 'magenta', 'cyan'
ag = np.linspace(0.5, 14, 300); cmap = cm.coolwarm; norm_c = colors.Normalize(-0.8, 0.4)

fig, ax = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={'width_ratios': [1, 1.5]}, constrained_layout=True)
s = base & np.isfinite(feh) & np.isfinite(mg)
h, xe, ye = np.histogram2d(feh[s], mg[s], bins=[80, 60], range=[(-1.5, 0.5), (-0.05, 0.45)])
him = np.full_like(h, np.nan); him[h > 0] = np.log10(h[h > 0])
ax[0].imshow(him.T, origin='lower', extent=[-1.5, 0.5, -0.05, 0.45], aspect='auto', cmap='Greys',
             vmin=np.nanpercentile(him, 3), vmax=np.nanpercentile(him, 99.5), zorder=0)
xx = np.linspace(-1.5, 0.5, 60)
ax[0].plot(xx, c.slope_acc2*xx + c.inter_acc2, 'k-', lw=1.2, zorder=2)
ax[0].plot(xx, divline(xx), 'g--', lw=1.6, zorder=3)
ax[0].scatter(feh[eos_hi], mg[eos_hi], s=16, c=CHI, edgecolors='k', linewidths=0.3, zorder=5, label=f'$\\alpha$-rich (n={int(eos_hi.sum())})')
ax[0].scatter(feh[eos_lo], mg[eos_lo], s=16, c=CLO, edgecolors='k', linewidths=0.3, zorder=5, label=f'$\\alpha$-poor (n={int(eos_lo.sum())})')
ax[0].set_xlim(-1.5, 0.5); ax[0].set_ylim(-0.05, 0.45); ax[0].legend(frameon=False, fontsize=9, loc='upper right')
label_axes(ax[0], '[Fe/H]', '[Mg/Fe]', 'Low-$\\alpha$ + two Eos branches')

edges = np.arange(-0.8, 0.4 + 1e-9, 0.1); EOS_LO, EOS_HI = -0.8, -0.3
ymax = 0
for i in range(len(edges)-1):
    b = disc & rel_ok & (feh >= edges[i]) & (feh < edges[i+1])
    if age[b].size >= 50: ymax = max(ymax, gaussian_kde(age[b])(ag).max())
for i in range(len(edges)-1):
    lo, hi = edges[i], edges[i+1]; fc = 0.5*(lo+hi)
    b = disc & rel_ok & (feh >= lo) & (feh < hi); y = age[b]
    if y.size >= 50:
        d2 = gaussian_kde(y)(ag); match = (fc >= EOS_LO) & (fc <= EOS_HI)
        ax[1].plot(ag, d2, color=cmap(norm_c(fc)), lw=4.5 if match else 1.3, alpha=1.0 if match else 0.6,
                   zorder=4 if match else 2, label=(f'disc {lo:.1f}<[Fe/H]<{hi:.1f}' if match else None))
        ax[1].plot([np.median(y)]*2, [1.05*ymax, 1.12*ymax], color=cmap(norm_c(fc)), lw=3, solid_capstyle='butt')
for sel, col, lab in [(eos_hi, CHI, r'Eos $\alpha$-rich'), (eos_lo, CLO, r'Eos $\alpha$-poor')]:
    y = age[sel & rel_ok]
    ax[1].plot(ag, gaussian_kde(y)(ag), color=col, lw=3.5, zorder=6, label=f'{lab} (n={y.size}, med={np.median(y):.1f})')
    ax[1].plot([np.median(y)]*2, [1.05*ymax, 1.12*ymax], color=col, lw=4, solid_capstyle='butt')
ax[1].set_xlim(0.5, 14); ax[1].set_ylim(0, 1.18*ymax)
label_axes(ax[1], 'age [Gyr] (Anders+2023 spAgeqrCal)', 'number density', r'Age distribution: low-$\alpha$ disc by [Fe/H] + Eos')
ax[1].legend(frameon=False, fontsize=9, loc='upper left')
sm = cm.ScalarMappable(norm=norm_c, cmap=cmap); sm.set_array([])
fig.colorbar(sm, ax=ax[1], pad=0.01).set_label('[Fe/H] of low-$\\alpha$ disc bin')
fig.savefig(FIG / '01_eos_age_dist_anders.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_eos_age_dist_anders.png')
for sel, lab in [(eos_hi, 'a-rich'), (eos_lo, 'a-poor'), (eos, 'both')]:
    y = age[sel & rel_ok]; print(f'  Eos {lab}: n={y.size} med={np.median(y):.1f}')
