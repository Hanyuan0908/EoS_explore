"""Age distribution of the low-alpha disc in [Fe/H] bins (KDE, coloured by [Fe/H]),
with the two Eos populations (alpha-rich / alpha-poor) overplotted for comparison.
Left: [Mg/Fe]-[Fe/H] with the low-alpha region and the two Eos branches. AstroNN ages.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy.stats import gaussian_kde
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
age = np.asarray(cat['age'], float); aerr = np.asarray(cat['age_model_error'], float)
base = np.asarray(m['base'], bool); thin_al = np.asarray(m['thin_al'], bool)
rel_ok = np.isfinite(age) & np.isfinite(aerr) & (aerr/age < 0.3)

# CANONICAL Eos cut (matches the bifurcation figure): Davies halo & low-alpha wedge & -0.9<[Fe/H]<-0.2
halo = base & ((ecc > 0.7) | (lz < 0))
def acc(f): return c.slope_acc*f + c.inter_acc
def hl(f): return c.slope_acc2*f + c.inter_acc2
def divline(f): return 0.317*f + 0.353
lowa = halo & (feh > -0.9) & (feh < -0.2) & (mg > acc(feh)) & (mg < hl(feh)) & (al > c.alfe_cut)
eos_hi = lowa & (mg > divline(feh)); eos_lo = lowa & (mg <= divline(feh))
disc = thin_al & (vphi > 150)
CHI, CLO = 'magenta', 'cyan'
ag = np.linspace(0.5, 14, 300)
cmap = cm.coolwarm; norm = colors.Normalize(-0.8, 0.4)

fig, ax = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={'width_ratios': [1, 1.5]}, constrained_layout=True)
# --- left: [Mg/Fe]-[Fe/H] + Eos branches ---
s = base & np.isfinite(feh) & np.isfinite(mg)
h, xe, ye = np.histogram2d(feh[s], mg[s], bins=[80, 60], range=[(-1.5, 0.5), (-0.05, 0.45)])
him = np.full_like(h, np.nan); him[h > 0] = np.log10(h[h > 0])
ax[0].imshow(him.T, origin='lower', extent=[-1.5, 0.5, -0.05, 0.45], aspect='auto', cmap='Greys',
             vmin=np.nanpercentile(him, 3), vmax=np.nanpercentile(him, 99.5), zorder=0)
xx = np.linspace(-1.5, 0.5, 60)
ax[0].plot(xx, c.slope_acc2*xx + c.inter_acc2, 'k-', lw=1.2, zorder=2)
ax[0].scatter(feh[eos_hi], mg[eos_hi], s=14, c=CHI, edgecolors='k', linewidths=0.3, zorder=5)
ax[0].scatter(feh[eos_lo], mg[eos_lo], s=14, c=CLO, edgecolors='k', linewidths=0.3, zorder=5)
ax[0].text(-0.3, 0.09, r'low-$\alpha$', color='0.3', fontsize=11, rotation=-15)
ax[0].set_xlim(-1.5, 0.5); ax[0].set_ylim(-0.05, 0.45)
label_axes(ax[0], '[Fe/H]', '[Mg/Fe]', 'Low-$\\alpha$ sequence + two Eos branches')
# --- right: age distributions per [Fe/H] bin (low-alpha disc) ---
edges = np.arange(-0.8, 0.4 + 1e-9, 0.1)
EOS_LO, EOS_HI = -0.8, -0.3          # metallicity range spanning the two Eos branches
for i in range(len(edges)-1):
    lo, hi = edges[i], edges[i+1]; fc = 0.5*(lo+hi)
    b = disc & rel_ok & (feh >= lo) & (feh < hi)
    y = age[b]
    if y.size >= 50:
        d = gaussian_kde(y)(ag)
        match = (fc >= EOS_LO) & (fc <= EOS_HI)      # same [Fe/H] as Eos -> thick
        ax[1].plot(ag, d, color=cmap(norm(fc)), lw=4.5 if match else 1.3,
                   alpha=1.0 if match else 0.65, zorder=4 if match else 2,
                   label=(f'low-$\\alpha$ disc, {lo:.1f}<[Fe/H]<{hi:.1f}' if match else None))
# median-age rug ticks (like reference)
ymax = 0
for i in range(len(edges)-1):
    lo, hi = edges[i], edges[i+1]; fc = 0.5*(lo+hi)
    b = disc & rel_ok & (feh >= lo) & (feh < hi); y = age[b]
    if y.size >= 50: ymax = max(ymax, gaussian_kde(y)(ag).max())
for i in range(len(edges)-1):
    lo, hi = edges[i], edges[i+1]; fc = 0.5*(lo+hi)
    b = disc & rel_ok & (feh >= lo) & (feh < hi); y = age[b]
    if y.size >= 50:
        ax[1].plot([np.median(y)]*2, [1.05*ymax, 1.12*ymax], color=cmap(norm(fc)), lw=3, solid_capstyle='butt')
# Eos branches
for sel, col, lab in [(eos_hi, CHI, r'Eos $\alpha$-rich'), (eos_lo, CLO, r'Eos $\alpha$-poor')]:
    y = age[sel & rel_ok]
    ax[1].plot(ag, gaussian_kde(y)(ag), color=col, lw=3.5, zorder=6,
               label=f'{lab} (n={y.size}, med={np.median(y):.1f})')
    ax[1].plot([np.median(y)]*2, [1.05*ymax, 1.12*ymax], color=col, lw=4, solid_capstyle='butt')
ax[1].set_xlim(0.5, 14); ax[1].set_ylim(0, 1.18*ymax)
label_axes(ax[1], 'age [Gyr] (AstroNN)', 'number density', r'Age distribution: low-$\alpha$ disc by [Fe/H] (colour) + Eos')
ax[1].legend(frameon=False, fontsize=10, loc='upper left')
sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
fig.colorbar(sm, ax=ax[1], pad=0.01).set_label('[Fe/H] of low-$\\alpha$ disc bin')
fig.savefig(FIG / '01_eos_age_dist.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_eos_age_dist.png')
for sel, lab in [(eos_hi, 'a-rich'), (eos_lo, 'a-poor')]:
    y = age[sel & rel_ok]; print(f'  Eos {lab}: n={y.size} median age={np.median(y):.1f} 16-84=[{np.percentile(y,16):.1f},{np.percentile(y,84):.1f}]')
