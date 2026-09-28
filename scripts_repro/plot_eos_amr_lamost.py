"""LAMOST (Xiang+24 MSTO ages) version of the age-metallicity contour plane
(companion to the anders/bingo panels from plot_eos_age_extcat.py).

Grey = base-sample density in [Fe/H] vs age; nested 90/60/30% KDE contours per
population: high-alpha disc, low-alpha disc, Splash, Eos. Population definitions are
identical to plot_eos_age_extcat.py (canonical Eos halo cut with the [Al/Fe]>-0.12
floor). Age quality: finite, 0<age<14 Gyr, sigma_age/age<0.3.

Data: data_repro/our_lamost_subgiant_ddpayne.fits.gz. Output figures_repro/01_eos_amr_lamost.png.
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
cat = load_catalog('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_lamost_subgiant_ddpayne.fits.gz')
m = make_masks(cat, c)
feh = np.asarray(cat['fe_h'], float); mg = np.asarray(cat['mg_fe'], float); vphi = np.asarray(cat['galvt'], float)
al = np.asarray(cat['al_fe'], float); lz = np.asarray(cat['lz'], float)
rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float); ecc = (rap - rperi)/(rap + rperi)
age = np.asarray(cat['age'], float); aerr = np.asarray(cat['age_model_error'], float)
base = np.asarray(m['base'], bool); thin_al = np.asarray(m['thin_al'], bool); thick_al = np.asarray(m['thick_al'], bool)
rel_ok = np.isfinite(age) & (age > 0) & (age < 14) & np.isfinite(aerr) & (aerr/age < 0.3)

halo = base & ((ecc > 0.7) | (lz < 0))
def acc(f): return c.slope_acc*f + c.inter_acc
def hl(f): return c.slope_acc2*f + c.inter_acc2
eos = halo & (feh > -0.9) & (feh < -0.2) & (mg > acc(feh)) & (mg < hl(feh)) & (al > c.alfe_cut)

LABEL = 'Xiang+24 MSTO'
pops = [('high-$\\alpha$ disc', thick_al & (vphi > 150), 'seagreen'),
        ('low-$\\alpha$ disc',  thin_al & (vphi > 150), 'royalblue'),
        ('Splash',              thick_al & (vphi < 80), 'darkorange'),
        ('Eos',                 eos, 'red')]
AGER = (0, 14); FEHR = (-1.15, 0.5)

fig, ax = plt.subplots(figsize=(9.5, 5.6), constrained_layout=True)
sb = base & rel_ok & np.isfinite(feh) & np.isfinite(age)
hb, xb, yb = np.histogram2d(age[sb], feh[sb], bins=[70, 70], range=[AGER, FEHR])
hbi = np.full_like(hb, np.nan); hbi[hb > 0] = np.log10(hb[hb > 0])
ax.imshow(hbi.T, origin='lower', extent=[*AGER, *FEHR], aspect='auto', cmap='Greys',
          alpha=0.6, vmin=np.nanpercentile(hbi, 3), vmax=np.nanpercentile(hbi, 99.5), zorder=0)
AG, FG = np.meshgrid(np.linspace(*AGER, 120), np.linspace(*FEHR, 120)); grid = np.vstack([AG.ravel(), FG.ravel()])
for lab, sel, col in pops:
    p2 = sel & rel_ok & np.isfinite(feh) & np.isfinite(age)
    xy = np.vstack([age[p2], feh[p2]]); kde = gaussian_kde(xy); dens = kde(xy)
    levels = sorted(np.percentile(dens, [10, 40, 70]))
    Z = kde(grid).reshape(AG.shape)
    ax.contour(AG, FG, Z, levels=levels, colors=[col], linewidths=[1.0, 1.6, 2.2], zorder=3)
    ax.plot([], [], color=col, lw=2.2, label=f'{lab} (n={int(p2.sum())})')
ax.set_xlim(*AGER); ax.set_ylim(*FEHR)
label_axes(ax, f'age [Gyr] ({LABEL})', '[Fe/H]', f'metallicity-age (90/60/30% contours;  $\\sigma_{{age}}/age<0.3$)')
ax.legend(frameon=False, fontsize=9.5, loc='lower left')
fig.savefig(FIG / '01_eos_amr_lamost.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_eos_amr_lamost.png')
for lab, sel, col in pops:
    print(f'  {lab}: n={int((sel & rel_ok).sum())}')
