"""APOGEE Eos age distributions across FOUR age catalogues (4 stacked panels):
  (1) AstroNN (Leung & Bovy 2019)   -- lite cache 'age'; sigma_age/age < ASIG
  (2) BINGO (Ciuca+2024)            -- age = 10^pred_logAge (the 'age' column);
                                       Ciuca+24 quality cuts. (BINGO has NO separate
                                       calibrated/uncalibrated age; age_lowess_correct
                                       in that file is ~AstroNN, so it is NOT used.)
  (3) Anders et al. 2023            -- spAgeCal; f_spAge clean & e_spAge/spAgeCal<AREL
  (4) Sanders & Das 2018            -- APOGEE-survey subset, coord-matched <1"

Each panel: Eos (solid red), Splash (solid orange), low-alpha disc (dashed blue), plus
the two Eos branches (faint dashed) split by the Davies line [Mg/Fe]=0.317[Fe/H]+0.353:
alpha-rich/metal-poor (above) and alpha-poor/metal-rich (below).

Populations (kinematic, APOGEE lite cache):
  Eos    = base & (e>0.7|Lz<0) & -0.9<[Fe/H]<-0.2 & Mg-wedge & [Al/Fe]>-0.12
  Splash = thick_al & Vphi<80 & [Fe/H]>-0.9 ;  disc = thin_al & Vphi>150
AstroNN/BINGO/Anders matched by APOGEE_ID; Sanders&Das by sky coordinates.
Run with 'strict' arg for tighter age-error cuts.
Output figures_repro/01_eos_age_dist_4cat_branches[_strict].png
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
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
warnings.filterwarnings('ignore')

REPO = '/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore'
CAT = '/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE'
sys.path.insert(0, REPO + '/eos-figures')
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 14, 'axes.labelsize': 18, 'legend.fontsize': 10.5,
    'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})
FIG = Path(REPO + '/figures_repro')
CEOS, CSPL, CDISC = '#E8112D', '#E8712B', '#1F6FB2'
CMP, CMR = '#7A0177', '#00868B'
c = Cuts()

LN10 = np.log(10)
STRICT = 'strict' in sys.argv
# absolute age-error mode: pass e.g. 'abs1.5' -> AstroNN & Anders use an ABSOLUTE Gyr cut.
ABSARG = [a for a in sys.argv if a.startswith('abs')]
ABSMODE = len(ABSARG) > 0
AABS = float(ABSARG[0][3:]) if (ABSMODE and ABSARG[0][3:]) else 1.5
# log-sigma mode: pass 'logsig' -> ALL THREE use the SAME sigma(log10 tau) < LSIG cut
# (BINGO natively via pred_logAge_std; AstroNN/Anders via sigma_tau/(tau*ln10)).
LOGMODE = 'logsig' in sys.argv
LSIG = 0.1
ASIG = 0.15 if STRICT else 0.3      # AstroNN sigma_age/age (relative mode)
BSTD = LSIG if LOGMODE else (0.10 if STRICT else 0.2)   # BINGO sigma(log age)
AREL = 0.20 if STRICT else 0.3      # Anders23 e_spAge/spAgeCal (relative mode)
SUF = ('_logsig' if LOGMODE else (f'_abs{AABS:g}' if ABSMODE else '') + ('_strict' if STRICT else ''))
print(('LOGSIG %.2f' % LSIG) if LOGMODE else (('ABS %.1f Gyr' % AABS) if ABSMODE else ('STRICT' if STRICT else 'default')),
      '| BINGO sig(logt)<=', BSTD)


def norm(a):
    return np.char.strip(np.array([(s.decode() if isinstance(s, bytes) else str(s)) for s in np.asarray(a)]))


# --- populations from lite cache ---
lite = load_catalog(REPO + '/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(lite, c)
feh = np.asarray(lite['fe_h'], float); mg = np.asarray(lite['mg_fe'], float); al = np.asarray(lite['al_fe'], float)
lz = np.asarray(lite['lz'], float); vphi = np.asarray(lite['galvt'], float)
rap = np.asarray(lite['rap'], float); rperi = np.asarray(lite['rperi'], float)
with np.errstate(invalid='ignore'):
    ecc = (rap - rperi) / (rap + rperi)
base = np.asarray(m['base'], bool); thick = np.asarray(m['thick_al'], bool); thin = np.asarray(m['thin_al'], bool)
halo = base & ((ecc > 0.7) | (lz < 0))
eos = halo & (feh > -0.9) & (feh < -0.2) & (mg > c.slope_acc * feh + c.inter_acc) \
    & (mg < c.slope_acc2 * feh + c.inter_acc2) & (al > c.alfe_cut)
divline = 0.317 * feh + 0.353
eos_mp = eos & (mg >= divline)     # alpha-rich / metal-poor branch
eos_mr = eos & (mg < divline)      # alpha-poor / metal-rich branch
spl = thick & (vphi < 80) & (feh > -0.9)
disc = thin & (vphi > 150)
apid = np.char.strip(np.asarray(lite['apogee_id']).astype(str))
print('Eos', int(eos.sum()), '| a-rich/metal-poor', int(eos_mp.sum()), '| a-poor/metal-rich', int(eos_mr.sum()))
POP = {'Eos': eos, 'mp': eos_mp, 'mr': eos_mr, 'Splash': spl, 'disc': disc}

# --- AstroNN (lite) ---
age_a = np.asarray(lite['age'], float); ae_a = np.asarray(lite['age_model_error'], float)
okA = np.isfinite(age_a) & (age_a > 0) & (age_a < 20) & np.isfinite(ae_a) \
    & ((ae_a / (age_a * LN10) < LSIG) if LOGMODE else (ae_a < AABS) if ABSMODE else (ae_a / age_a < ASIG))
astronn = {k: age_a[v & okA] for k, v in POP.items()}

# --- BINGO = 10^pred_logAge (the 'age' column), Ciuca+24 quality; by APOGEE_ID ---
b = Table.read(CAT + '/APOGEE_DR17_bingoages.fits')
apb = np.char.strip(np.asarray(b['APOGEE_ID']).astype(str))
bage = np.asarray(b['age'], float)          # == 10**pred_logAge (verified exactly)
std = np.asarray(b['pred_logAge_std'], float)
qb = (np.asarray(b['MG_FE_FLAG']) == 0) & (np.asarray(b['FE_H_FLAG']) == 0) \
    & (np.asarray(b['SNR'], float) > 100) & (np.asarray(b['TEFF_1'], float) > 4000) & (np.asarray(b['TEFF_1'], float) < 5500) \
    & (np.asarray(b['LOGG_1'], float) > 1) & (np.asarray(b['LOGG_1'], float) < 3.5) & (std <= BSTD) \
    & (np.asarray(b['FE_H_1'], float) > -1) & ~((bage < 8) & (np.asarray(b['MG_FE_1'], float) > 0.2))
bidx = {a: i for i, a in enumerate(apb)}


def bmatch(mask):
    idx = np.array([bidx[a] for a in apid[mask] if a in bidx], int)
    idx = idx[qb[idx]]
    v = bage[idx]
    return v[np.isfinite(v) & (v > 0)]


bingo = {k: bmatch(v) for k, v in POP.items()}

# --- Anders et al. 2023: spAgeCal, f_spAge clean & rel err cut; by APOGEE id ---
an = fits.open(CAT + '/Anders23_Age.fit')[1].data
aap = norm(an['APOGEE']); aage = np.asarray(an['spAgeCal'], float); aerr = np.asarray(an['e_spAge'], float)
aflag = np.array([str(x).strip() for x in an['f_spAge']])
qa = (aflag == '') & np.isfinite(aage) & (aage > 0) & np.isfinite(aerr) \
    & ((aerr / (aage * LN10) < LSIG) if LOGMODE else (aerr < AABS) if ABSMODE else (aerr / aage < AREL))
aidx = {a: i for i, a in enumerate(aap)}


def amatch(mask):
    idx = np.array([aidx[a] for a in apid[mask] if a in aidx], int)
    idx = idx[qa[idx]]
    return aage[idx]


anders = {k: amatch(v) for k, v in POP.items()}

# --- Sanders & Das 2018 (APOGEE survey), coordinate match <1" (no error column) ---
sd = fits.open(CAT + '/Sanders_Das_age.fits')[1].data
surv = norm(sd['survey']); apo = np.char.find(surv, 'APOGEE') >= 0
csd = SkyCoord(np.asarray(sd['ra'], float)[apo] * u.deg, np.asarray(sd['dec'], float)[apo] * u.deg)
sdage = np.asarray(sd['age'], float)[apo]
clite = SkyCoord(np.asarray(lite['ra'], float) * u.deg, np.asarray(lite['dec'], float) * u.deg)
ii, sep, _ = clite.match_to_catalog_sky(csd)
sd_age_lite = np.where(sep.arcsec < 1.0, sdage[ii], np.nan)
okSD = np.isfinite(sd_age_lite) & (sd_age_lite > 0) & (sd_age_lite < 14)
sd = {k: sd_age_lite[v & okSD] for k, v in POP.items()}

if LOGMODE:
    lblA = fr'AstroNN  ($\sigma_{{\log\tau}}<{LSIG:g}$)'
    lblAn = fr'Anders+2023 (spAgeCal; clean, $\sigma_{{\log\tau}}<{LSIG:g}$)'
elif ABSMODE:
    lblA = fr'AstroNN  ($\sigma_\tau<{AABS:g}$ Gyr)'
    lblAn = fr'Anders+2023 (spAgeCal; clean, $\sigma_\tau<{AABS:g}$ Gyr)'
else:
    lblA = fr'AstroNN  ($\sigma_\tau/\tau<{ASIG}$)'
    lblAn = fr'Anders+2023 (spAgeCal; clean, $\sigma_\tau/\tau<{AREL}$)'
PANELS = [(lblA, astronn, 14),
          (f'BINGO ($10^{{\\rm pred\\_logAge}}$; Ciuca+24 cuts, $\\sigma_{{\\log\\tau}}\\leq{BSTD}$)', bingo, 18),
          (lblAn, anders, 14),
          ('Sanders \\& Das 2018 (APOGEE; no age error avail.)', sd, 14)]


def kde(ax, a, col, ls, lw, lab, xg, alpha=1.0):
    a = a[np.isfinite(a)]
    if a.size < 8:
        return
    k = gaussian_kde(a)
    ax.plot(xg, k(xg), color=col, ls=ls, lw=lw, alpha=alpha,
            label=fr'{lab} ($n={a.size}$, $\tilde\tau={np.median(a):.1f}$)')


fig, axes = plt.subplots(4, 1, figsize=(8.6, 13.6), sharex=False, constrained_layout=True)
for ax, (label, data, xmax) in zip(axes, PANELS):
    xg = np.linspace(0, xmax, 450)
    kde(ax, data['Eos'], CEOS, '-', 2.8, 'Eos', xg)
    kde(ax, data['mp'], CMP, (0, (5, 2)), 1.7, r'Eos $\alpha$-rich/metal-poor', xg, alpha=0.85)
    kde(ax, data['mr'], CMR, (0, (5, 2)), 1.7, r'Eos $\alpha$-poor/metal-rich', xg, alpha=0.85)
    kde(ax, data['Splash'], CSPL, '-', 2.6, 'Splash', xg)
    kde(ax, data['disc'], CDISC, '--', 2.6, r'low-$\alpha$ disc', xg)
    ax.set_xlim(0, xmax); ax.set_ylim(bottom=0); ax.set_ylabel('density')
    ax.text(0.025, 0.94, label, transform=ax.transAxes, va='top', ha='left', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', ncol=1)
    ax.set_xlabel('age [Gyr]')
fig.savefig(FIG / f'01_eos_age_dist_4cat_branches{SUF}.png', bbox_inches='tight')
print('wrote', FIG / f'01_eos_age_dist_4cat_branches{SUF}.png')
for label, data, _ in PANELS:
    print(label.split('(')[0].strip(), {k: (v.size, round(float(np.median(v)), 1)) for k, v in data.items() if v.size})
