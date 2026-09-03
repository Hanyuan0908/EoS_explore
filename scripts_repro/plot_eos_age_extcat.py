"""Eos age test with an EXTERNAL spectroscopic-age catalogue (Anders 2023 or BINGO/Ciuca 2024),
matched by APOGEE_ID. Two figures:
  *_agedist : combined-Eos age distribution vs matched-metallicity low-alpha disc bins.
  *_amr     : age-metallicity plane, nested contours (high/low-a disc, Splash, Eos).
QUALITY CUT (both catalogues, applied to Eos AND disc): finite, 0<age<20 Gyr, and sigma_age/age < 0.3.
Usage: python plot_eos_age_extcat.py [anders|bingo]
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy.stats import gaussian_kde
from astropy.io import fits
REPO = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/eos-figures')
sys.path.insert(0, str(REPO))
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
from eos_figures.plotting import label_axes
c = Cuts()
FIG = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/figures_repro')
APO = '/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE'
CAT = sys.argv[1] if len(sys.argv) > 1 else 'anders'
SPECS = {
    'anders': (f'{APO}/APOGEE_AstroNNdist_Anders23age_BJdist.fits', 'APOGEE_ID_2', 'spAgeqrCal', 'e_spAgeqrCal', 'Anders+2023 spAgeqrCal'),
    'bingo':  (f'{APO}/APOGEE_DR17_bingoages.fits', 'APOGEE_ID', 'age_lowess_correct', 'age_total_error', 'BINGO (Ciucã+2024)'),
}
FN, IDCOL, AGECOL, ERRCOL, LABEL = SPECS[CAT]

cat = load_catalog(f'/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c)
feh = np.asarray(cat['fe_h'], float); mg = np.asarray(cat['mg_fe'], float); vphi = np.asarray(cat['galvt'], float)
al = np.asarray(cat['al_fe'], float); lz = np.asarray(cat['lz'], float)
rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float); ecc = (rap - rperi)/(rap + rperi)
aid = np.asarray(cat['apogee_id']); base = np.asarray(m['base'], bool); thin_al = np.asarray(m['thin_al'], bool); thick_al = np.asarray(m['thick_al'], bool)

d = fits.open(FN)[1].data
def norm(a): return np.array([(s.decode() if isinstance(s, bytes) else str(s)).strip() for s in np.asarray(a)])
did = norm(d[IDCOL]); o = np.argsort(did); dids = did[o]
p = np.clip(np.searchsorted(dids, aid), 0, len(dids)-1); ok = dids[p] == aid; src = o[p]
age = np.where(ok, np.asarray(d[AGECOL], float)[src], np.nan)
aerr = np.where(ok, np.asarray(d[ERRCOL], float)[src], np.nan)
rel_ok = np.isfinite(age) & (age > 0) & (age < 20) & np.isfinite(aerr) & (aerr/age < 0.3)   # QUALITY CUT

halo = base & ((ecc > 0.7) | (lz < 0))
def acc(f): return c.slope_acc*f + c.inter_acc
def hl(f): return c.slope_acc2*f + c.inter_acc2
def divline(f): return 0.317*f + 0.353
eos = halo & (feh > -0.9) & (feh < -0.2) & (mg > acc(feh)) & (mg < hl(feh)) & (al > c.alfe_cut)
eos_hi = eos & (mg > divline(feh)); eos_lo = eos & (mg <= divline(feh))
disc = thin_al & (vphi > 150)
ag = np.linspace(0.5, 14, 300); cmap = cm.coolwarm; norm_c = colors.Normalize(-0.8, 0.4)

# ============ (A) combined-Eos age distribution vs matched-metallicity disc ============
fig, ax = plt.subplots(figsize=(8.6, 5.4), constrained_layout=True)
edges = np.arange(-0.8, 0.4 + 1e-9, 0.1); EOS_LO, EOS_HI = -0.8, -0.3
ymax = 0
for i in range(len(edges)-1):
    b = disc & rel_ok & (feh >= edges[i]) & (feh < edges[i+1])
    if age[b].size >= 50: ymax = max(ymax, gaussian_kde(age[b])(ag).max())
ye = age[eos & rel_ok]; ymax = max(ymax, gaussian_kde(ye)(ag).max())
for i in range(len(edges)-1):
    lo, hi = edges[i], edges[i+1]; fc = 0.5*(lo+hi)
    b = disc & rel_ok & (feh >= lo) & (feh < hi); y = age[b]
    if y.size >= 50:
        match = (fc >= EOS_LO) & (fc <= EOS_HI)
        ax.plot(ag, gaussian_kde(y)(ag), color=cmap(norm_c(fc)), lw=4.5 if match else 1.3,
                alpha=1.0 if match else 0.55, zorder=4 if match else 2,
                label=(f'low-$\\alpha$ disc {lo:.1f}<[Fe/H]<{hi:.1f}' if match else None))
ax.plot(ag, gaussian_kde(ye)(ag), color='red', lw=4.0, zorder=6, label=f'Eos (n={ye.size}, med={np.median(ye):.1f})')
ax.axvline(np.median(ye), color='red', ls=':', lw=1.5)
ax.set_xlim(0.5, 14); ax.set_ylim(0, 1.1*ymax)
label_axes(ax, f'age [Gyr] ({LABEL})', 'number density',
           f'Eos vs matched-metallicity disc  ({LABEL};  $\\sigma_{{age}}/age<0.3$)')
ax.legend(frameon=False, fontsize=9, loc='upper left')
fig.savefig(FIG / f'01_eos_age_dist_{CAT}.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / f'01_eos_age_dist_{CAT}.png')

# ============ (B) age-metallicity plane ============
pops = [('high-$\\alpha$ disc', thick_al & (vphi > 150), 'seagreen'),
        ('low-$\\alpha$ disc',  thin_al & (vphi > 150), 'royalblue'),
        ('Splash',              thick_al & (vphi < 80), 'darkorange'),
        ('Eos',                 eos, 'red')]
AGER = (0, 13.5); FEHR = (-1.15, 0.5)
fig2, ax2 = plt.subplots(figsize=(9.5, 5.6), constrained_layout=True)
sb = base & rel_ok & np.isfinite(feh) & np.isfinite(age)
hb, xb, yb = np.histogram2d(age[sb], feh[sb], bins=[70, 70], range=[AGER, FEHR])
hbi = np.full_like(hb, np.nan); hbi[hb > 0] = np.log10(hb[hb > 0])
ax2.imshow(hbi.T, origin='lower', extent=[*AGER, *FEHR], aspect='auto', cmap='Greys',
           alpha=0.6, vmin=np.nanpercentile(hbi, 3), vmax=np.nanpercentile(hbi, 99.5), zorder=0)
AG, FG = np.meshgrid(np.linspace(*AGER, 120), np.linspace(*FEHR, 120)); grid = np.vstack([AG.ravel(), FG.ravel()])
for lab, sel, col in pops:
    p2 = sel & rel_ok & np.isfinite(feh) & np.isfinite(age)
    xy = np.vstack([age[p2], feh[p2]]); kde = gaussian_kde(xy); dens = kde(xy)
    levels = sorted(np.percentile(dens, [10, 40, 70]))
    Z = kde(grid).reshape(AG.shape)
    ax2.contour(AG, FG, Z, levels=levels, colors=[col], linewidths=[1.0, 1.6, 2.2], zorder=3)
    ax2.plot([], [], color=col, lw=2.2, label=f'{lab} (n={int(p2.sum())})')
ax2.set_xlim(*AGER); ax2.set_ylim(*FEHR)
label_axes(ax2, f'age [Gyr] ({LABEL})', '[Fe/H]', f'metallicity-age (90/60/30% contours;  $\\sigma_{{age}}/age<0.3$)')
ax2.legend(frameon=False, fontsize=9.5, loc='lower left')
fig2.savefig(FIG / f'01_eos_amr_{CAT}.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / f'01_eos_amr_{CAT}.png')
print(f'[{CAT}] Eos combined n={ye.size} med={np.median(ye):.1f};  branches (thin): a-rich n={(eos_hi&rel_ok).sum()} med={np.median(age[eos_hi&rel_ok]):.1f}, a-poor n={(eos_lo&rel_ok).sum()} med={np.median(age[eos_lo&rel_ok]):.1f}')
for a0,b0 in [(-0.8,-0.7),(-0.7,-0.6),(-0.6,-0.5),(-0.5,-0.4)]:
    y=age[disc&rel_ok&(feh>=a0)&(feh<b0)]; print(f'  disc {a0}..{b0}: n={y.size} med={np.median(y):.1f}')
