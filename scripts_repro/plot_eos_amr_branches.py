"""Age-metallicity plane (4 population contours over density) with the Eos stars overplotted
as scatter, colour-coded into the two Eos branches (alpha-rich upper vs alpha-poor lower,
split by the Davies divider [Mg/Fe]=0.317[Fe/H]+0.353). Left panel: same split shown in
[Fe/H]-[Mg/Fe]. AstroNN ages.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
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
age = np.asarray(cat['age'], float); aerr = np.asarray(cat['age_model_error'], float)
base = np.asarray(m['base'], bool); thin_al = np.asarray(m['thin_al'], bool); thick_al = np.asarray(m['thick_al'], bool)
rel_ok = np.isfinite(age) & np.isfinite(aerr) & (aerr/age < 0.3)

# CANONICAL Eos cut: Davies halo & low-alpha wedge & -0.9<[Fe/H]<-0.2
al = np.asarray(cat['al_fe'], float); lz = np.asarray(cat['lz'], float)
rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float); ecc = (rap - rperi)/(rap + rperi)
halo = base & ((ecc > 0.7) | (lz < 0))
def acc(f): return c.slope_acc*f + c.inter_acc
def hl(f): return c.slope_acc2*f + c.inter_acc2
def divline(f): return 0.317*f + 0.353
eos = halo & (feh > -0.9) & (feh < -0.2) & (mg > acc(feh)) & (mg < hl(feh)) & (al > c.alfe_cut)
eos_hi = eos & (mg > divline(feh))    # alpha-rich (upper branch)
eos_lo = eos & (mg <= divline(feh))   # alpha-poor (lower branch)
CHI, CLO = 'magenta', 'cyan'

pops = [('high-$\\alpha$ disc', thick_al & (vphi > 150), 'seagreen'),
        ('low-$\\alpha$ disc',  thin_al & (vphi > 150), 'royalblue'),
        ('Splash',              thick_al & (vphi < 80), 'darkorange'),
        ('Eos',                 eos, 'red')]
AGER = (0, 13.5); FEHR = (-1.15, 0.5)

fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.4), gridspec_kw={'width_ratios': [1, 1.35]}, constrained_layout=True)
# --- left: [Fe/H]-[Mg/Fe], cut + divider + the two Eos branches ---
s = base & np.isfinite(feh) & np.isfinite(mg)
h, xe, ye = np.histogram2d(feh[s], mg[s], bins=[80, 60], range=[(-1.6, 0.5), (-0.05, 0.45)])
him = np.full_like(h, np.nan); him[h > 0] = np.log10(h[h > 0])
ax[0].imshow(him.T, origin='lower', extent=[-1.6, 0.5, -0.05, 0.45], aspect='auto', cmap='Greys',
             vmin=np.nanpercentile(him, 3), vmax=np.nanpercentile(him, 99.5), zorder=0)
xx = np.linspace(-1.6, 0.5, 60)
ax[0].plot(xx, c.slope_acc2*xx + c.inter_acc2, 'r-', lw=1.6, zorder=2)
ax[0].plot(xx, c.slope_acc*xx + c.inter_acc, 'r--', lw=1.0, zorder=2)
xd = np.linspace(-0.9, -0.2, 40); ax[0].plot(xd, divline(xd), color='k', ls=':', lw=1.8, zorder=4)
ax[0].scatter(feh[eos_hi], mg[eos_hi], s=16, c=CHI, edgecolors='k', linewidths=0.3, zorder=5)
ax[0].scatter(feh[eos_lo], mg[eos_lo], s=16, c=CLO, edgecolors='k', linewidths=0.3, zorder=5)
ax[0].text(-0.55, 0.31, r'high-$\alpha$', color='seagreen', fontsize=10, fontweight='bold')
ax[0].text(-0.32, 0.10, r'low-$\alpha$', color='royalblue', fontsize=10, fontweight='bold', rotation=-15)
ax[0].set_xlim(-1.6, 0.5); ax[0].set_ylim(-0.05, 0.45)
label_axes(ax[0], '[Fe/H]', '[Mg/Fe]', 'Eos split by Davies divider (dotted)')
# --- right: contours over density + Eos scatter by branch ---
sb = base & rel_ok & np.isfinite(feh) & np.isfinite(age)
hb, xb, yb = np.histogram2d(age[sb], feh[sb], bins=[70, 70], range=[AGER, FEHR])
hbi = np.full_like(hb, np.nan); hbi[hb > 0] = np.log10(hb[hb > 0])
ax[1].imshow(hbi.T, origin='lower', extent=[*AGER, *FEHR], aspect='auto', cmap='Greys',
             alpha=0.55, vmin=np.nanpercentile(hbi, 3), vmax=np.nanpercentile(hbi, 99.5), zorder=0)
AG, FG = np.meshgrid(np.linspace(*AGER, 120), np.linspace(*FEHR, 120)); grid = np.vstack([AG.ravel(), FG.ravel()])
for lab, sel, col in pops:
    p = sel & rel_ok & np.isfinite(feh) & np.isfinite(age)
    xy = np.vstack([age[p], feh[p]]); kde = gaussian_kde(xy); dens = kde(xy)
    Z = kde(grid).reshape(AG.shape)
    ax[1].contour(AG, FG, Z, levels=sorted(np.percentile(dens, [10, 40, 70])), colors=[col], linewidths=[0.9, 1.4, 2.0], zorder=2)
    ax[1].plot([], [], color=col, lw=2, label=f'{lab}')
for sel, col, lab in [(eos_hi, CHI, r'Eos $\alpha$-rich (upper)'), (eos_lo, CLO, r'Eos $\alpha$-poor (lower)')]:
    p = sel & rel_ok
    ax[1].scatter(age[p], feh[p], s=26, c=col, edgecolors='k', linewidths=0.4, zorder=5,
                  label=f'{lab} (n={int(p.sum())}, med age={np.median(age[p]):.1f})')
ax[1].set_xlim(*AGER); ax[1].set_ylim(*FEHR)
label_axes(ax[1], 'age [Gyr] (AstroNN)', '[Fe/H]', 'metallicity-age: contours + Eos branches')
ax[1].legend(frameon=False, fontsize=8.5, loc='lower left', ncol=1)
fig.suptitle('Two Eos populations on the age-metallicity plane (AstroNN ages)', fontsize=12)
fig.savefig(FIG / '01_eos_amr_branches.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_eos_amr_branches.png')
for sel, lab in [(eos_hi, 'alpha-rich (upper)'), (eos_lo, 'alpha-poor (lower)')]:
    p = sel & rel_ok
    print(f'  Eos {lab:20s}: n={int(p.sum())}  median age={np.median(age[p]):.1f}  [Fe/H] med={np.median(feh[p]):+.2f}  [Mg/Fe] med={np.median(mg[p]):.2f}')
