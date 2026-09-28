"""Publication figure: Eos age distributions in AstroNN and BINGO (2 stacked panels).
Populations: Eos (solid red) + its two branches split by the Davies line
[Mg/Fe]=0.317[Fe/H]+0.353 (alpha-rich/metal-poor, alpha-poor/metal-rich; dashed),
Splash (solid orange), low-alpha disc (dashed blue).
Quality cuts (not annotated on the figure): AstroNN sigma_age/age<0.15;
BINGO age=10^pred_logAge, Ciuca+24 cuts with sigma(log tau)<=0.1.
ONE shared legend (identical for both panels), placed above the panels; no n counts.
Output figures_repro/01_eos_age_dist_pub.{png,pdf}
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
sys.path.insert(0, REPO + '/eos-figures')
from eos_figures.data import load_catalog, make_masks
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
# observational population palette (CONVENTIONS_observational.md)
CEOS = '#E8112D'   # Eos (whole)      -- red
CMP = '#FF6347'    # Eos metal-poor   (alpha-rich, upper) -- tomato
CMR = '#1F6FB2'    # Eos metal-rich   (alpha-poor, lower) -- blue
CSPL = '#E8712B'   # Splash           -- orange
CDISC = '#2B2B2B'  # low-alpha disc   -- near-black
c = Cuts()

# --- populations ---
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
dl = 0.317 * feh + 0.353
eos_mp = eos & (mg >= dl); eos_mr = eos & (mg < dl)
spl = thick & (vphi < 80) & (feh > -0.9); disc = thin & (vphi > 150)
apid = np.char.strip(np.asarray(lite['apogee_id']).astype(str))
POP = {'Eos': eos, 'mp': eos_mp, 'mr': eos_mr, 'Splash': spl, 'disc': disc}

# --- AstroNN (strict: sigma_age/age < 0.15) ---
age_a = np.asarray(lite['age'], float); ae_a = np.asarray(lite['age_model_error'], float)
okA = np.isfinite(age_a) & (age_a > 0) & (age_a < 20) & (ae_a / age_a < 0.15)
astronn = {k: age_a[v & okA] for k, v in POP.items()}

# --- BINGO = 10^pred_logAge, Ciuca+24 cuts, sigma(log tau) <= 0.1 ---
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
    xg = np.linspace(0, xmax, 500)
    handles = []
    for (lab, col, ls, lw, a), key in zip(SER, KEY):
        d = data[key]; d = d[np.isfinite(d)]
        k = gaussian_kde(d)
        h, = ax.plot(xg, k(xg), color=col, ls=ls, lw=lw, alpha=a, label=lab)
        handles.append(h)
    ax.set_xlim(0, xmax); ax.set_ylim(bottom=0)
    ax.set_ylabel('density')
    ax.minorticks_on()
    return handles


handles = draw(ax1, astronn, 14)
draw(ax2, bingo, 18)
ax1.set_xlabel('age [Gyr]'); ax2.set_xlabel('age [Gyr]')
ax1.text(0.03, 0.94, '(a) AstroNN', transform=ax1.transAxes, va='top', ha='left', fontsize=16, fontweight='bold')
ax2.text(0.03, 0.94, '(b) BINGO', transform=ax2.transAxes, va='top', ha='left', fontsize=16, fontweight='bold')

# one shared legend (identical for both panels) above the figure -- no counts, no titles
fig.legend(handles, [s[0] for s in SER], loc='lower center', bbox_to_anchor=(0.5, 1.005),
           ncol=3, frameon=False, handlelength=2.6, columnspacing=1.8, fontsize=13)

fig.savefig(REPO + '/figures_repro/01_eos_age_dist_pub.png', bbox_inches='tight')
fig.savefig(REPO + '/figures_repro/01_eos_age_dist_pub.pdf', bbox_inches='tight')
print('wrote 01_eos_age_dist_pub.png/.pdf')
for nm, data in [('AstroNN', astronn), ('BINGO', bingo)]:
    print(nm, {k: (data[k].size, round(float(np.median(data[k])), 1)) for k in KEY})
