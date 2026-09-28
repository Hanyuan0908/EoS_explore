"""Non-publish: age distributions of Eos / high-alpha Splash / low-alpha disc,
compared across three age catalogues (three stacked panels, top to bottom):
  (top)    AstroNN                         -- quality sigma_age/age < 0.3
  (middle) BINGO under Ciuca+24 cuts       -- SNR>100, 4000<Teff<5500, 1<logg<3.5,
                                              sigma(log tau)<=0.2, [Fe/H]>-1, binary
                                              filter; age = age_lowess_correct (calibrated)
  (bottom) LAMOST Xiang isochronal         -- absolute age error < 1 Gyr

Populations (same cuts in every panel):
  Eos    = canonical halo cut & low-a wedge, -0.9<[Fe/H]<-0.2
  Splash = thick_al & V_phi<80 & [Fe/H]>-0.9   (paper def)
  disc   = thin_al & V_phi>150                  (low-alpha disc)
Eos/Splash/disc membership uses kinematics: from the lite cache for AstroNN & BINGO
(BINGO matched by APOGEE_ID), and from the LAMOST cache for the LAMOST panel.

BINGO uses the CALIBRATED lowess age (not the raw 10**pred_logAge) so the three
panels share a physical 0-14 Gyr scale. Data (Mac-only): APOGEE_DR17_bingoages.fits
+ data_repro caches. Output figures_repro/01_age_dist_3cat.png.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
import warnings
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
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
    'font.size': 14, 'axes.labelsize': 18, 'legend.fontsize': 12.5,
    'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})
FIG = Path(REPO + '/figures_repro')
CEOS, CSPL, CDISC = '#E8112D', '#E8712B', '#1F6FB2'
c = Cuts()


def pops(cat):
    m = make_masks(cat, c)
    feh = np.asarray(cat['fe_h'], float); mg = np.asarray(cat['mg_fe'], float); al = np.asarray(cat['al_fe'], float)
    lz = np.asarray(cat['lz'], float); vphi = np.asarray(cat['galvt'], float)
    rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float); ecc = (rap - rperi) / (rap + rperi)
    base = np.asarray(m['base'], bool); thick = np.asarray(m['thick_al'], bool); thin = np.asarray(m['thin_al'], bool)
    halo = base & ((ecc > 0.7) | (lz < 0))
    eos = halo & (feh > -0.9) & (feh < -0.2) & (mg > c.slope_acc * feh + c.inter_acc) \
        & (mg < c.slope_acc2 * feh + c.inter_acc2) & (al > c.alfe_cut)
    spl = thick & (vphi < 80) & (feh > -0.9)
    disc = thin & (vphi > 150)
    return eos, spl, disc


# --- AstroNN (lite cache) ---
lite = load_catalog(REPO + '/data_repro/our_apogee_dr17_lite_ann.fits.gz')
eos_l, spl_l, disc_l = pops(lite)
age_a = np.asarray(lite['age'], float); ae_a = np.asarray(lite['age_model_error'], float)
rok_a = np.isfinite(age_a) & (age_a > 0) & (age_a < 20) & (ae_a / age_a < 0.3)
apid_lite = np.char.strip(np.asarray(lite['apogee_id']).astype(str))
astronn = {'Eos': age_a[eos_l & rok_a], 'Splash': age_a[spl_l & rok_a], 'disc': age_a[disc_l & rok_a]}

# --- LAMOST Xiang isochronal (its own cache) ---
lam = load_catalog(REPO + '/data_repro/our_lamost_subgiant_ddpayne.fits.gz')
eL, sL, dL = pops(lam)
aL = np.asarray(lam['age'], float); aeL = np.asarray(lam['age_model_error'], float)
rL = np.isfinite(aL) & (aL > 0) & (aL < 14) & (aeL < 1.0)     # absolute error < 1 Gyr
lamost = {'Eos': aL[eL & rL], 'Splash': aL[sL & rL], 'disc': aL[dL & rL]}

# --- BINGO under Ciuca+24 cuts (match lite membership by APOGEE_ID) ---
b = Table.read('/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE/APOGEE_DR17_bingoages.fits')
apb = np.char.strip(np.asarray(b['APOGEE_ID']).astype(str))
alow = np.asarray(b['age_lowess_correct'], float); std = np.asarray(b['pred_logAge_std'], float)
qual = (np.asarray(b['MG_FE_FLAG']) == 0) & (np.asarray(b['FE_H_FLAG']) == 0) & np.isfinite(alow) \
    & (np.asarray(b['SNR'], float) > 100) & (np.asarray(b['TEFF_1'], float) > 4000) & (np.asarray(b['TEFF_1'], float) < 5500) \
    & (np.asarray(b['LOGG_1'], float) > 1) & (np.asarray(b['LOGG_1'], float) < 3.5) & (std <= 0.2) \
    & (np.asarray(b['FE_H_1'], float) > -1) & ~((alow < 8) & (np.asarray(b['MG_FE_1'], float) > 0.2))
bidx = {a: i for i, a in enumerate(apb)}


def bmatch(mask):
    idx = np.array([bidx[a] for a in apid_lite[mask] if a in bidx], int)
    return alow[idx[qual[idx]]]


bingo = {'Eos': bmatch(eos_l), 'Splash': bmatch(spl_l), 'disc': bmatch(disc_l)}

PANELS = [('AstroNN', astronn),
          ('BINGO (Ciuca+24 cuts, calibrated)', bingo),
          ('LAMOST isochronal ($\\sigma_\\tau<1$ Gyr)', lamost)]
SER = [('Eos', CEOS, '-'), ('Splash', CSPL, '-'), ('low-$\\alpha$ disc', CDISC, '--')]
KEY = {'Eos': 'Eos', 'Splash': 'Splash', 'low-$\\alpha$ disc': 'disc'}
xg = np.linspace(0, 14, 400)

fig, axes = plt.subplots(3, 1, figsize=(8.4, 10.2), sharex=True, constrained_layout=True)
for ax, (label, data) in zip(axes, PANELS):
    for lab, col, ls in SER:
        a = data[KEY[lab]]; a = a[np.isfinite(a)]
        if a.size < 8:
            continue
        kde = gaussian_kde(a)
        ax.plot(xg, kde(xg), color=col, ls=ls, lw=2.6,
                label=fr'{lab} ($n={a.size}$, $\tilde\tau={np.median(a):.1f}$)')
    ax.set_xlim(0, 14); ax.set_ylim(bottom=0)
    ax.set_ylabel('density')
    ax.text(0.025, 0.93, label, transform=ax.transAxes, va='top', ha='left', fontsize=15, fontweight='bold')
    ax.legend(loc='upper right')
axes[-1].set_xlabel('age [Gyr]')

fig.savefig(FIG / '01_age_dist_3cat.png', bbox_inches='tight')
print('wrote', FIG / '01_age_dist_3cat.png')
for label, data in PANELS:
    print(label, {k: (v.size, round(float(np.median(v)), 1)) for k, v in data.items() if v.size})
