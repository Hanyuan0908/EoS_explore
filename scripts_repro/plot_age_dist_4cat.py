"""Non-publish: age distributions of Eos / high-alpha Splash / low-alpha disc,
compared across FOUR age catalogues (four stacked panels, top to bottom):
  (1) AstroNN                         -- quality sigma_age/age < 0.3
  (2) BINGO under Ciuca+24 cuts       -- SNR>100, 4000<Teff<5500, 1<logg<3.5,
                                         sigma(log tau)<=0.2, [Fe/H]>-1, binary
                                         filter; age = age_lowess_correct (calibrated)
  (3) LAMOST Xiang isochronal         -- absolute age error < 1 Gyr (independent
                                         LAMOST selection: see note below)
  (4) DistMass (Stone-Martinez+2024)  -- spectro-photometric giant ages, matched to
                                         our APOGEE stars by APOGEE_ID; AGE_COR_SS
                                         (corrected), converted from yr to Gyr.

Populations (same kinematic cuts in every panel; from the lite cache except the
LAMOST panel which uses its own cache):
  Eos    = canonical halo cut & low-a wedge, -0.9<[Fe/H]<-0.2
  Splash = thick_al & V_phi<80 & [Fe/H]>-0.9   (paper def)
  disc   = thin_al & V_phi>150                  (low-alpha disc)

NOTE on LAMOST: the LAMOST panel selects Eos *within* LAMOST. A direct cross-match of
the APOGEE-selected Eos into Xiang's LAMOST catalogue yields ~0 stars, because our
APOGEE sample is giants (logg<3) while Xiang's LAMOST sample is subgiants (logg~3.5-4);
the two are disjoint stellar populations. DistMass, being APOGEE-based, matches 352/353
of the APOGEE Eos directly. StarHorse DR19/DR3 v2 has no age column (mass/dist/AV only).

Data (Mac-only): APOGEE_DR17_bingoages.fits, APOGEE_DistMass-DR17_v1.6.1.fits, caches.
Output figures_repro/01_age_dist_4cat.png.
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


def norm(a):
    return np.char.strip(np.array([(s.decode() if isinstance(s, bytes) else str(s)) for s in np.asarray(a)]))


def pops(cat):
    m = make_masks(cat, c)
    feh = np.asarray(cat['fe_h'], float); mg = np.asarray(cat['mg_fe'], float); al = np.asarray(cat['al_fe'], float)
    lz = np.asarray(cat['lz'], float); vphi = np.asarray(cat['galvt'], float)
    rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float)
    with np.errstate(invalid='ignore'):
        ecc = (rap - rperi) / (rap + rperi)
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

# --- LAMOST Xiang isochronal (independent selection in its own cache) ---
lam = load_catalog(REPO + '/data_repro/our_lamost_subgiant_ddpayne.fits.gz')
eL, sL, dL = pops(lam)
aL = np.asarray(lam['age'], float); aeL = np.asarray(lam['age_model_error'], float)
rL = np.isfinite(aL) & (aL > 0) & (aL < 14) & (aeL < 1.0)
lamost = {'Eos': aL[eL & rL], 'Splash': aL[sL & rL], 'disc': aL[dL & rL]}

# --- BINGO under Ciuca+24 cuts (match lite membership by APOGEE_ID) ---
b = Table.read(CAT + '/APOGEE_DR17_bingoages.fits')
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

# --- DistMass (Stone-Martinez+2024): match lite membership by APOGEE_ID; yr -> Gyr ---
# Quality cuts: valid asymmetric error, BITMASK bit-2 clear (star INSIDE the mass-model
# training parameter space; Table 3: value 2 = "outside covered parameter space"), and
# age error < 1.5 Gyr. AGE_ERR is in dex (log10 age), so convert to Gyr per star:
# the fixed 1.5 Gyr cut therefore scales as ~age*0.35 and preferentially removes OLD
# stars, biasing the survivors young (Eos median 5.5 -> 3.6, n 94 -> 18). Caveat noted.
dm = fits.open(CAT + '/APOGEE_DistMass-DR17_v1.6.1.fits')[1].data
dap = norm(dm['APOGEE_ID'])
dage = np.asarray(dm['AGE_COR_SS'], float) / 1e9          # corrected, self-consistent grid
dbm = np.asarray(dm['BITMASK'], np.int64)
dae = np.asarray(dm['AGE_ERR'], float)                    # (N,2) = [+dex, -dex]
d_valid = (dae[:, 0] > 0) & (dae[:, 1] < 0)               # a real error was estimated
d_err_gyr = 0.5 * (dage * 10 ** dae[:, 0] - dage * 10 ** dae[:, 1])  # sym 1-sigma in Gyr
d_ok = np.isfinite(dage) & (dage > 0) & (dage < 20) & d_valid \
    & ((dbm & 2) == 0) & (d_err_gyr < 1.5)
didx = {a: i for i, a in enumerate(dap)}


def dmatch(mask):
    idx = np.array([didx[a] for a in apid_lite[mask] if a in didx], int)
    idx = idx[d_ok[idx]]
    return dage[idx]


distmass = {'Eos': dmatch(eos_l), 'Splash': dmatch(spl_l), 'disc': dmatch(disc_l)}

PANELS = [('AstroNN', astronn),
          ('BINGO (Ciuca+24 cuts, calibrated)', bingo),
          ('LAMOST isochronal ($\\sigma_\\tau<1$ Gyr)', lamost),
          ('DistMass (S-M+24, in-space, $\\sigma_\\tau<1.5$ Gyr)', distmass)]
SER = [('Eos', CEOS, '-'), ('Splash', CSPL, '-'), ('low-$\\alpha$ disc', CDISC, '--')]
KEY = {'Eos': 'Eos', 'Splash': 'Splash', 'low-$\\alpha$ disc': 'disc'}
xg = np.linspace(0, 14, 400)

fig, axes = plt.subplots(4, 1, figsize=(8.4, 13.2), sharex=True, constrained_layout=True)
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

fig.savefig(FIG / '01_age_dist_4cat.png', bbox_inches='tight')
print('wrote', FIG / '01_age_dist_4cat.png')
for label, data in PANELS:
    print(label, {k: (v.size, round(float(np.median(v)), 1)) for k, v in data.items() if v.size})
