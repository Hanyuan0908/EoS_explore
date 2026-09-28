"""Non-publish: our Eos and high-alpha Splash selections highlighted on the
age-metallicity plane using Xiang & Rix LAMOST subgiant (isochronal/MSTO) ages.

Companion to plot_eos_splash_bingo_amr.py, but with Xiang LAMOST ages (physical,
to ~14 Gyr) instead of raw BINGO ages -- and no cross-match is needed because the
LAMOST subgiant cache carries kinematics, DD-Payne [Fe/H]/[Mg/Fe]/[Al/Fe] and the
Xiang ages together.

Background: LAMOST disc sample (grey, [Fe/H]>-1) in [Fe/H] vs tau (x reversed).
Overplotted:
  - Eos            (red)    -- canonical halo cut & low-a wedge, -0.9<[Fe/H]<-0.2
  - high-a Splash  (orange) -- thick_al & V_phi<80
Age quality: finite, 0<age<14 Gyr, sigma_age/age<0.3.

Data: data_repro/our_lamost_subgiant_ddpayne.fits.gz. Output figures_repro/01_eos_splash_lamost_amr.png.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

REPO = '/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore'
sys.path.insert(0, REPO + '/eos-figures')
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 15, 'axes.labelsize': 19, 'legend.fontsize': 14,
    'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': True,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})
FIG = Path(REPO + '/figures_repro')
CEOS, CSPL = '#E8112D', '#E8712B'

c = Cuts()
cat = load_catalog(REPO + '/data_repro/our_lamost_subgiant_ddpayne.fits.gz')
m = make_masks(cat, c)
feh = np.asarray(cat['fe_h'], float); mg = np.asarray(cat['mg_fe'], float); al = np.asarray(cat['al_fe'], float)
lz = np.asarray(cat['lz'], float); vphi = np.asarray(cat['galvt'], float)
rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float)
ecc = (rap - rperi) / (rap + rperi)
age = np.asarray(cat['age'], float); aerr = np.asarray(cat['age_model_error'], float)
rel_ok = np.isfinite(age) & (age > 0) & (age < 14) & np.isfinite(aerr) & (aerr / age < 0.3)

base = np.asarray(m['base'], bool); thick_al = np.asarray(m['thick_al'], bool)
halo = base & ((ecc > 0.7) | (lz < 0))
eos = halo & (feh > -0.9) & (feh < -0.2) & (mg > c.slope_acc * feh + c.inter_acc) \
    & (mg < c.slope_acc2 * feh + c.inter_acc2) & (al > c.alfe_cut) & rel_ok
splash = thick_al & (vphi < 80) & (feh > -0.9) & rel_ok   # paper def: high-a, [Fe/H]>-0.9, V_phi<80
disc = base & rel_ok & (feh > -1)                        # grey background (disc regime)

AGER = (0, 14); FEHR = (-1.15, 0.75)
fig, ax = plt.subplots(figsize=(10.2, 7.2), constrained_layout=True)

ax.scatter(age[disc], feh[disc], s=3, c='0.72', edgecolors='none', alpha=0.30, rasterized=True, zorder=1)
for xv in (10, 12):
    ax.axvline(xv, color='0.4', ls='--', lw=1.0, zorder=2)

ax.scatter(age[splash], feh[splash], s=16, c=CSPL, edgecolors='none', alpha=0.75, rasterized=True, zorder=3,
           label=fr'high-$\alpha$ Splash ($n={int(splash.sum())}$)')
ax.scatter(age[eos], feh[eos], s=34, c=CEOS, edgecolors='k', linewidths=0.4, alpha=0.95, zorder=5,
           label=fr'Eos ($n={int(eos.sum())}$)')

ax.set_xlim(AGER[1], AGER[0]); ax.set_ylim(*FEHR)
ax.set_xlabel(r'$\tau$ [Gyr]  (Xiang LAMOST)')
ax.set_ylabel('[Fe/H] [dex]')
ax.legend(loc='lower right', framealpha=0.9)
ax.text(0.03, 0.05, 'grey: LAMOST disc sample', transform=ax.transAxes,
        fontsize=12, color='0.35', va='bottom')

fig.savefig(FIG / '01_eos_splash_lamost_amr.png', bbox_inches='tight')
print('wrote', FIG / '01_eos_splash_lamost_amr.png')
print(f'Eos    n={int(eos.sum())}  median tau={np.median(age[eos]):.1f}, [Fe/H]={np.median(feh[eos]):+.2f}')
print(f'Splash n={int(splash.sum())}  median tau={np.median(age[splash]):.1f}, [Fe/H]={np.median(feh[splash]):+.2f}')
