"""Metallicity-age plane with ANDERS+2023 spectroscopic ages (spAgeqrCal, matched by APOGEE_ID).
Same as plot_eos_amr.py but the age axis is Anders 2023 instead of AstroNN (less saturated at old age).
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from astropy.io import fits
REPO = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/eos-figures')
sys.path.insert(0, str(REPO))
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
from eos_figures.plotting import label_axes
c = Cuts()
FIG = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/figures_repro')
cat = load_catalog('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c)
feh = np.asarray(cat['fe_h'], float); mg = np.asarray(cat['mg_fe'], float); vphi = np.asarray(cat['galvt'], float)
base = np.asarray(m['base'], bool); thin_al = np.asarray(m['thin_al'], bool); thick_al = np.asarray(m['thick_al'], bool)
al = np.asarray(cat['al_fe'], float); lz = np.asarray(cat['lz'], float)
rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float); ecc = (rap - rperi)/(rap + rperi)
aid = np.asarray(cat['apogee_id'])

# --- ANDERS 2023 ages by APOGEE_ID ---
d = fits.open('/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE/APOGEE_AstroNNdist_Anders23age_BJdist.fits')[1].data
def norm(a): return np.array([(s.decode() if isinstance(s, bytes) else str(s)).strip() for s in np.asarray(a)])
did = norm(d['APOGEE_ID_2']); o = np.argsort(did); dids = did[o]
p = np.clip(np.searchsorted(dids, aid), 0, len(dids)-1); ok = dids[p] == aid; src = o[p]
age = np.where(ok, np.asarray(d['spAgeqrCal'], float)[src], np.nan)
rel_ok = np.isfinite(age) & (age > 0) & (age < 20)

halo = base & ((ecc > 0.7) | (lz < 0))
def acc(f): return c.slope_acc*f + c.inter_acc
def hl(f): return c.slope_acc2*f + c.inter_acc2
eos_sel = halo & (feh > -0.9) & (feh < -0.2) & (mg > acc(feh)) & (mg < hl(feh)) & (al > c.alfe_cut)
pops = [('high-$\\alpha$ disc', thick_al & (vphi > 150), 'seagreen'),
        ('low-$\\alpha$ disc',  thin_al & (vphi > 150), 'royalblue'),
        ('Splash',              thick_al & (vphi < 80), 'darkorange'),
        ('Eos',                 eos_sel, 'red')]
AGER = (0, 13.5); FEHR = (-1.15, 0.5)

fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.4), gridspec_kw={'width_ratios': [1, 1.35]}, constrained_layout=True)
s = base & np.isfinite(feh) & np.isfinite(mg)
h, xe, ye = np.histogram2d(feh[s], mg[s], bins=[80, 60], range=[(-1.6, 0.5), (-0.05, 0.45)])
him = np.full_like(h, np.nan); him[h > 0] = np.log10(h[h > 0])
ax[0].imshow(him.T, origin='lower', extent=[-1.6, 0.5, -0.05, 0.45], aspect='auto', cmap='Greys',
             vmin=np.nanpercentile(him, 3), vmax=np.nanpercentile(him, 99.5), zorder=0)
xx = np.linspace(-1.6, 0.5, 60)
ax[0].plot(xx, c.slope_acc2*xx + c.inter_acc2, 'r-', lw=2, zorder=3)
ax[0].plot(xx, c.slope_acc*xx + c.inter_acc, 'r--', lw=1.2, zorder=3)
ax[0].text(-0.55, 0.31, r'high-$\alpha$', color='seagreen', fontsize=11, fontweight='bold')
ax[0].text(-0.35, 0.10, r'low-$\alpha$', color='royalblue', fontsize=11, fontweight='bold', rotation=-15)
ax[0].text(-1.45, 0.02, 'accreted', color='0.3', fontsize=9)
ax[0].set_xlim(-1.6, 0.5); ax[0].set_ylim(-0.05, 0.45)
label_axes(ax[0], '[Fe/H]', '[Mg/Fe]', r'high/low-$\alpha$ cut (solid = split)')

sb = base & rel_ok & np.isfinite(feh) & np.isfinite(age)
hb, xb, yb = np.histogram2d(age[sb], feh[sb], bins=[70, 70], range=[AGER, FEHR])
hbi = np.full_like(hb, np.nan); hbi[hb > 0] = np.log10(hb[hb > 0])
ax[1].imshow(hbi.T, origin='lower', extent=[*AGER, *FEHR], aspect='auto', cmap='Greys',
             alpha=0.6, vmin=np.nanpercentile(hbi, 3), vmax=np.nanpercentile(hbi, 99.5), zorder=0)
AG, FG = np.meshgrid(np.linspace(*AGER, 120), np.linspace(*FEHR, 120))
grid = np.vstack([AG.ravel(), FG.ravel()])
ENC = [0.9, 0.6, 0.3]
for lab, sel, col in pops:
    p2 = sel & rel_ok & np.isfinite(feh) & np.isfinite(age)
    xy = np.vstack([age[p2], feh[p2]]); kde = gaussian_kde(xy); dens = kde(xy)
    levels = sorted(np.percentile(dens, [100*(1-f) for f in ENC]))
    Z = kde(grid).reshape(AG.shape)
    ax[1].contour(AG, FG, Z, levels=levels, colors=[col], linewidths=[1.0, 1.6, 2.2], zorder=3)
    ax[1].plot([], [], color=col, lw=2.2, label=f'{lab} (n={int(p2.sum())})')
ax[1].set_xlim(*AGER); ax[1].set_ylim(*FEHR)
label_axes(ax[1], 'age [Gyr] (Anders+2023 spAgeqrCal)', '[Fe/H]', 'metallicity-age (nested 90/60/30% contours over density)')
ax[1].legend(frameon=False, fontsize=9.5, loc='lower left')
fig.suptitle('Eos metallicity-age structure vs high/low-$\\alpha$ disc & Splash (Anders 2023 ages)', fontsize=12)
fig.savefig(FIG / '01_eos_amr_anders.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_eos_amr_anders.png')
