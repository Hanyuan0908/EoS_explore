"""Non-publish: our Eos and high-alpha Splash selections highlighted on the
Ciuca+24-style age-metallicity plane (BINGO ages).

Background: the Ciuca+24 disc sample (grey, R>4, same quality cuts) in the
[Fe/H] vs tau plane (raw BINGO age, reversed x). Overplotted:
  - Eos            (red)    -- canonical halo cut & low-a wedge, -0.9<[Fe/H]<-0.2
  - high-a Splash  (orange) -- thick_al & V_phi<80
Both selected on the kinematic lite cache and matched to BINGO by APOGEE_ID;
age quality sigma(log tau)<=0.2. Same age scale (raw BINGO) as Ciuca Fig 3 so the
two populations sit on the same plane -- these raw ages are not physical absolute
ages (see plot_ciuca_amr_bingo.py).

Data (Mac-only): APOGEE/APOGEE_DR17_bingoages.fits + data_repro lite cache.
Output figures_repro/01_eos_splash_bingo_amr.png.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
import warnings
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.table import Table
warnings.filterwarnings('ignore')

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
BINGO = '/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE/APOGEE_DR17_bingoages.fits'
CEOS, CSPL = '#E8112D', '#E8712B'

# --- our Eos / Splash selection on the kinematic lite cache ---
c = Cuts()
lite = load_catalog(REPO + '/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(lite, c)
feh = np.asarray(lite['fe_h'], float); mg = np.asarray(lite['mg_fe'], float); al = np.asarray(lite['al_fe'], float)
lz = np.asarray(lite['lz'], float); vphi = np.asarray(lite['galvt'], float)
rap = np.asarray(lite['rap'], float); rperi = np.asarray(lite['rperi'], float)
ecc = (rap - rperi) / (rap + rperi)
base = np.asarray(m['base'], bool); thick_al = np.asarray(m['thick_al'], bool)
halo = base & ((ecc > 0.7) | (lz < 0))
eos = halo & (feh > -0.9) & (feh < -0.2) & (mg > c.slope_acc * feh + c.inter_acc) \
    & (mg < c.slope_acc2 * feh + c.inter_acc2) & (al > c.alfe_cut)
splash = thick_al & (vphi < 80) & (feh > -0.9)   # paper Splash: high-a, [Fe/H]>-0.9, V_phi<80
apid_lite = np.char.strip(np.asarray(lite['apogee_id']).astype(str))

# --- BINGO catalogue: ages + Ciuca+24 quality mask ---
b = Table.read(BINGO)
apid_b = np.char.strip(np.asarray(b['APOGEE_ID']).astype(str))
bage = np.asarray(b['age'], float)                       # raw BINGO age (Ciuca Fig 3 axis)
bfeh = np.asarray(b['FE_H_1'], float)
bstd = np.asarray(b['pred_logAge_std'], float)
qual = (np.asarray(b['MG_FE_FLAG']) == 0) & (np.asarray(b['FE_H_FLAG']) == 0) & np.isfinite(bage) & np.isfinite(bfeh) \
    & (np.asarray(b['SNR'], float) > 100) & (np.asarray(b['TEFF_1'], float) > 4000) & (np.asarray(b['TEFF_1'], float) < 5500) \
    & (np.asarray(b['LOGG_1'], float) > 1) & (np.asarray(b['LOGG_1'], float) < 3.5) & (bstd <= 0.2) & (bfeh > -1) \
    & ~((bage < 8) & (np.asarray(b['MG_FE_1'], float) > 0.2))
bidx = {a: i for i, a in enumerate(apid_b)}


def match(mask):
    idx = np.array([bidx[a] for a in apid_lite[mask] if a in bidx], int)
    idx = idx[np.isfinite(bage[idx]) & np.isfinite(bstd[idx]) & (bstd[idx] <= 0.2)]
    return bage[idx], bfeh[idx]


eos_age, eos_feh = match(eos)
spl_age, spl_feh = match(splash)

AGER = (0, 18); FEHR = (-1.15, 0.75)
fig, ax = plt.subplots(figsize=(10.2, 7.2), constrained_layout=True)

# background: Ciuca disc sample (grey)
ax.scatter(bage[qual], bfeh[qual], s=3, c='0.72', edgecolors='none', alpha=0.30, rasterized=True, zorder=1)
for xv in (10, 12):
    ax.axvline(xv, color='0.4', ls='--', lw=1.0, zorder=2)

ax.scatter(spl_age, spl_feh, s=16, c=CSPL, edgecolors='none', alpha=0.75, rasterized=True, zorder=3,
           label=fr'high-$\alpha$ Splash ($n={spl_age.size}$)')
ax.scatter(eos_age, eos_feh, s=34, c=CEOS, edgecolors='k', linewidths=0.4, alpha=0.95, zorder=5,
           label=fr'Eos ($n={eos_age.size}$)')

ax.set_xlim(AGER[1], AGER[0]); ax.set_ylim(*FEHR)
ax.set_xlabel(r'$\tau$ [Gyr]  (BINGO)')
ax.set_ylabel('[Fe/H] [dex]')
ax.legend(loc='lower right', framealpha=0.9)
ax.text(0.03, 0.05, 'grey: disc sample (Ciuca+24 cuts)', transform=ax.transAxes,
        fontsize=12, color='0.35', va='bottom')

fig.savefig(FIG / '01_eos_splash_bingo_amr.png', bbox_inches='tight')
print('wrote', FIG / '01_eos_splash_bingo_amr.png')
print(f'Eos with BINGO age: {eos_age.size}  (median tau={np.median(eos_age):.1f}, [Fe/H]={np.median(eos_feh):+.2f})')
print(f'Splash with BINGO age: {spl_age.size}  (median tau={np.median(spl_age):.1f}, [Fe/H]={np.median(spl_feh):+.2f})')
