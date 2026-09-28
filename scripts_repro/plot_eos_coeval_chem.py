"""Chemistry of stars COEVAL with Eos.

For each age catalogue (BINGO calibrated / AstroNN):
  1. select Eos (canonical cut) and find the MODE of its age distribution (KDE peak);
  2. select all in-situ Milky Way stars with age within +-0.5 Gyr of that mode;
  3. contour those coeval stars in the [Mg/Fe]-[Fe/H] plane (over the grey in-situ
     density), with the Eos stars overplotted for reference.

Milky Way (in-situ) = base & [Al/Fe]>-0.12 (drops accreted). Eos = base & (e>0.7|Lz<0)
& -0.9<[Fe/H]<-0.2 & low-a wedge & [Al/Fe]>-0.12. BINGO age = age_lowess_correct with
sigma(log tau)<=0.2; AstroNN age with sigma_age/age<0.3. Two panels.
Output figures_repro/01_eos_coeval_chem.png.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from astropy.io import fits
REPO = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/eos-figures')
sys.path.insert(0, str(REPO))
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 15, 'axes.labelsize': 18, 'legend.fontsize': 12,
    'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 130, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})
FIG = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/figures_repro')
c = Cuts()
cat = load_catalog('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c); base = np.asarray(m['base'], bool)
feh = np.asarray(cat['fe_h'], float); mg = np.asarray(cat['mg_fe'], float); al = np.asarray(cat['al_fe'], float)
lz = np.asarray(cat['lz'], float); rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float)
with np.errstate(invalid='ignore'):
    ecc = (rap - rperi) / (rap + rperi)
aid = np.char.strip(np.asarray(cat['apogee_id']).astype(str))
halo = base & ((ecc > 0.7) | (lz < 0))
eos = halo & (feh > -0.9) & (feh < -0.2) & (mg > c.slope_acc*feh + c.inter_acc) \
    & (mg < c.slope_acc2*feh + c.inter_acc2) & (al > c.alfe_cut)
mw = base & (al > c.alfe_cut)                                   # in-situ Milky Way

# --- AstroNN ages (in lite cache) ---
aA = np.asarray(cat['age'], float); aAe = np.asarray(cat['age_model_error'], float)
okA = np.isfinite(aA) & (aA > 0) & (aA < 14) & np.isfinite(aAe) & (aAe/aA < 0.3)

# --- BINGO ages (matched by APOGEE_ID) ---
b = fits.open('/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE/APOGEE_DR17_bingoages.fits')[1].data
def norm(a): return np.char.strip(np.array([(s.decode() if isinstance(s, bytes) else str(s)) for s in np.asarray(a)]))
bid = norm(b['APOGEE_ID']); o = np.argsort(bid); bid_s = bid[o]
p = np.clip(np.searchsorted(bid_s, aid), 0, len(bid_s)-1); okid = bid_s[p] == aid; src = o[p]
bage = np.where(okid, np.asarray(b['age_lowess_correct'], float)[src], np.nan)   # calibrated (LOWESS -> seismic scale)
braw = np.where(okid, np.asarray(b['age'], float)[src], np.nan)                   # uncalibrated raw = 10**pred_logAge
bstd = np.where(okid, np.asarray(b['pred_logAge_std'], float)[src], np.nan)
okB = okid & np.isfinite(bage) & (bage > 0) & np.isfinite(bstd) & (bstd <= 0.2)
okBr = okid & np.isfinite(braw) & (braw > 0) & np.isfinite(bstd) & (bstd <= 0.2)

XR = (-1.7, 0.5); YR = (-0.1, 0.5)
xg = np.linspace(0, 20, 600)


def mode_of(ages):
    a = ages[np.isfinite(ages)]
    k = gaussian_kde(a); return xg[np.argmax(k(xg))]


def panel(ax, agecol, okage, label):
    emode = mode_of(agecol[eos & okage])
    coeval = mw & okage & (np.abs(agecol - emode) < 0.5)
    # grey in-situ MW density
    s = mw & np.isfinite(feh) & np.isfinite(mg)
    H, xe, ye = np.histogram2d(feh[s], mg[s], bins=[90, 60], range=[XR, YR])
    Hi = np.full_like(H, np.nan); Hi[H > 0] = np.log10(H[H > 0])
    ax.imshow(Hi.T, origin='lower', extent=[*XR, *YR], aspect='auto', cmap='Greys',
              alpha=0.55, vmin=np.nanpercentile(Hi, 3), vmax=np.nanpercentile(Hi, 99), zorder=0)
    # coeval contour
    Hc, xc, yc = np.histogram2d(feh[coeval], mg[coeval], bins=[45, 32], range=[XR, YR])
    Hs = gaussian_filter(Hc, 1.0)
    flat = np.sort(Hs.ravel())[::-1]; cs = np.cumsum(flat)/flat.sum()
    lv = sorted(set(flat[np.searchsorted(cs, f)] for f in (0.9, 0.6, 0.3)))
    xcc = 0.5*(xc[:-1]+xc[1:]); ycc = 0.5*(yc[:-1]+yc[1:])
    ax.contour(xcc, ycc, Hs.T, levels=lv, colors='#1F6FB2', linewidths=[1.2, 1.9, 2.6], zorder=4)
    ax.plot([], [], color='#1F6FB2', lw=2.2, label=fr'coeval MW ($|\tau-{emode:.1f}|<0.5$, n={int(coeval.sum())})')
    # Eos stars
    ax.scatter(feh[eos & okage], mg[eos & okage], s=16, c='#E8112D', edgecolors='k', linewidths=0.3,
               zorder=6, label=f'Eos (n={int((eos & okage).sum())})')
    ax.set_xlim(*XR); ax.set_ylim(*YR); ax.set_xlabel('[Fe/H]'); ax.set_ylabel('[Mg/Fe]')
    ax.set_title(fr'{label}: Eos age mode $\tau={emode:.1f}$ Gyr', fontsize=15)
    ax.legend(loc='upper right')
    print(f'{label}: Eos age mode={emode:.2f} Gyr; Eos n={int((eos&okage).sum())}; coeval MW n={int(coeval.sum())}')


fig, ax = plt.subplots(1, 3, figsize=(21, 6.3), constrained_layout=True)
panel(ax[0], braw, okBr, 'BINGO (uncalibrated, raw)')
panel(ax[1], bage, okB, 'BINGO (calibrated)')
panel(ax[2], aA, okA, 'AstroNN')
fig.savefig(FIG / '01_eos_coeval_chem.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_eos_coeval_chem.png')
