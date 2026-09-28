"""[McMillan17 series] Eos age distributions in AstroNN + BINGO, 2 stacked panels.
Populations from the FROM-SCRATCH kinematics (BJ21 photogeometric distance + Gaia DR3
astrometry & RV -> AGAMA + McMillan17), via scripts_repro/eos_mcm.py. Ages are the
same catalogues as before, matched by APOGEE_ID: AstroNN (lite cache, sigma_age/age<0.15);
BINGO 10^pred_logAge, Ciuca+24 cuts with sigma(log tau)<=0.1.
Output Fig_paper_mcmillan17/obs_eos_age_dist.{pdf,png}
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from astropy.table import Table
warnings.filterwarnings('ignore')

REPO = '/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore'
CAT = '/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE'
sys.path.insert(0, REPO + '/scripts_repro'); sys.path.insert(0, REPO + '/eos-figures')
from eos_mcm import load_mcm, make_masks_mcm, eos_split
from eos_figures.config import Cuts

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 14, 'axes.labelsize': 18, 'legend.fontsize': 13,
    'xtick.labelsize': 15, 'ytick.labelsize': 15,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5, 'xtick.minor.size': 3, 'ytick.minor.size': 3,
    'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})
OUT = REPO + '/Fig_paper_mcmillan17'
CEOS, CMP, CMR, CSPL, CDISC = '#E8112D', '#FF6347', '#1F6FB2', '#E8712B', '#2B2B2B'
c = Cuts()

cat = load_mcm(); m = make_masks_mcm(cat, c)
eos, mp, mr = eos_split(cat, m, c)
vt = np.asarray(cat['galvt'], float); feh = np.asarray(cat['fe_h'], float)
spl = np.asarray(m['thick_al'], bool) & (vt < 80) & (feh > -0.9)
disc = np.asarray(m['thin_al'], bool) & (vt > 150)
apid = np.char.strip(np.asarray(cat['apogee_id']).astype(str))
POP = {'Eos': eos, 'mp': mp, 'mr': mr, 'Splash': spl, 'disc': disc}

# --- AstroNN ages, matched by APOGEE_ID from the lite cache ---
lite = Table.read(REPO + '/data_repro/our_apogee_dr17_lite_ann.fits.gz')
lap = np.char.strip(np.asarray(lite['apogee_id']).astype(str))
la = {a: (ag, ae) for a, ag, ae in zip(lap, np.asarray(lite['age'], float), np.asarray(lite['age_model_error'], float))}
aA = np.array([la.get(a, (np.nan, np.nan))[0] for a in apid])
aeA = np.array([la.get(a, (np.nan, np.nan))[1] for a in apid])
okA = np.isfinite(aA) & (aA > 0) & (aA < 20) & (aeA / aA < 0.15)
astronn = {k: aA[v & okA] for k, v in POP.items()}

# --- BINGO 10^pred_logAge, Ciuca+24 cuts, sigma(log tau)<=0.1 ---
b = Table.read(CAT + '/APOGEE_DR17_bingoages.fits')
apb = np.char.strip(np.asarray(b['APOGEE_ID']).astype(str))
bage = np.asarray(b['age'], float); std = np.asarray(b['pred_logAge_std'], float)
qb = (np.asarray(b['MG_FE_FLAG']) == 0) & (np.asarray(b['FE_H_FLAG']) == 0) \
    & (np.asarray(b['SNR'], float) > 100) & (np.asarray(b['TEFF_1'], float) > 4000) & (np.asarray(b['TEFF_1'], float) < 5500) \
    & (np.asarray(b['LOGG_1'], float) > 1) & (np.asarray(b['LOGG_1'], float) < 3.5) & (std <= 0.1) \
    & (np.asarray(b['FE_H_1'], float) > -1) & ~((bage < 8) & (np.asarray(b['MG_FE_1'], float) > 0.2))
bidx = {a: i for i, a in enumerate(apb)}


def bmatch(mask):
    idx = np.array([bidx[a] for a in apid[mask] if a in bidx], int)
    idx = idx[qb[idx]]; v = bage[idx]
    return v[np.isfinite(v) & (v > 0)]


bingo = {k: bmatch(v) for k, v in POP.items()}

SER = [('Eos', CEOS, '-', 2.8, 1.0),
       ('Eos, metal-poor', CMP, (0, (5, 2)), 2.1, 1.0),
       ('Eos, metal-rich', CMR, (0, (5, 2)), 2.1, 1.0),
       ('Splash', CSPL, '-', 2.6, 1.0),
       (r'low-$\alpha$ disc', CDISC, (0, (7, 3)), 2.4, 1.0)]
KEY = ['Eos', 'mp', 'mr', 'Splash', 'disc']
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 9.0), constrained_layout=True)


def draw(ax, data, xmax):
    xg = np.linspace(0, xmax, 500); handles = []
    for (lab, col, ls, lw, a), key in zip(SER, KEY):
        d = data[key]; d = d[np.isfinite(d)]
        h, = ax.plot(xg, gaussian_kde(d)(xg), color=col, ls=ls, lw=lw, alpha=a, label=lab)
        handles.append(h)
    ax.set_xlim(0, xmax); ax.set_ylim(bottom=0); ax.set_ylabel('density'); ax.minorticks_on()
    return handles


handles = draw(ax1, astronn, 14); draw(ax2, bingo, 18)
ax1.set_xlabel('age [Gyr]'); ax2.set_xlabel('age [Gyr]')
ax1.text(0.03, 0.94, '(a) AstroNN', transform=ax1.transAxes, va='top', ha='left', fontsize=16, fontweight='bold')
ax2.text(0.03, 0.94, '(b) BINGO', transform=ax2.transAxes, va='top', ha='left', fontsize=16, fontweight='bold')
fig.legend(handles, [s[0] for s in SER], loc='lower center', bbox_to_anchor=(0.5, 1.005),
           ncol=3, frameon=False, handlelength=2.6, columnspacing=1.8, fontsize=13)
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/obs_eos_age_dist.{ext}', bbox_inches='tight')
print('wrote', OUT + '/obs_eos_age_dist.{pdf,png}')
for nm, data in [('AstroNN', astronn), ('BINGO', bingo)]:
    print(nm, {k: (data[k].size, round(float(np.median(data[k])), 1)) for k in KEY if data[k].size})
