"""Davies+2025 test: do the two Eos branches differ in RADIAL ACTION J_R?
Compute actions (J_R, J_z, L_z) with AGAMA's ActionFinder in the McMillan (2017) potential,
using full 6D phase space (R,phi,z,vR,vt,vz) matched from the AstroNN VAC by APOGEE_ID.
Eos = CANONICAL cut (Davies halo & low-alpha wedge & -0.9<[Fe/H]<-0.2), split by the divider.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, ks_2samp, mannwhitneyu
from astropy.io import fits
import agama
REPO = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/eos-figures')
sys.path.insert(0, str(REPO))
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
from eos_figures.plotting import label_axes
c = Cuts()
FIG = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/figures_repro')

# --- AGAMA setup: units (kpc, km/s, Msun) + McMillan 2017 potential ---
agama.setUnits(mass=1, length=1, velocity=1)
POT_INI = os.path.join(os.path.dirname(agama.__file__), 'data', 'McMillan17.ini')
pot = agama.Potential(POT_INI)
af = agama.ActionFinder(pot)
R0 = 8.2
vc = (-R0 * pot.force(R0, 0, 0)[0])**0.5
print(f'McMillan17 loaded; v_circ(R={R0})={vc:.1f} km/s (sanity ~233)')

# --- catalogue + canonical Eos cut ---
cat = load_catalog('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c)
feh = np.asarray(cat['fe_h'], float); mg = np.asarray(cat['mg_fe'], float)
al = np.asarray(cat['al_fe'], float); lz = np.asarray(cat['lz'], float)
rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float); ecc = (rap - rperi)/(rap + rperi)
aid = np.asarray(cat['apogee_id']); base = np.asarray(m['base'], bool); thin_al = np.asarray(m['thin_al'], bool)
halo = base & ((ecc > 0.7) | (lz < 0))
def acc(f): return c.slope_acc*f + c.inter_acc
def hl(f): return c.slope_acc2*f + c.inter_acc2
def divline(f): return 0.317*f + 0.353

# --- 6D phase space from AstroNN VAC (match by APOGEE_ID) ---
ann = fits.open('/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE/apogee_astroNN-DR17.fits')[1].data
def norm(a): return np.array([(s.decode() if isinstance(s, bytes) else str(s)).strip() for s in np.asarray(a)])
nid = norm(ann['APOGEE_ID']); o = np.argsort(nid); nid_s = nid[o]
p = np.clip(np.searchsorted(nid_s, aid), 0, len(nid_s)-1); ok = nid_s[p] == aid; src = o[p]
cc = lambda n: np.where(ok, np.asarray(ann[n], float)[src], np.nan)
R, phi, zz = cc('galr'), cc('galphi'), cc('galz'); vR, vt, vz = cc('galvr'), cc('galvt'), cc('galvz')
x = R*np.cos(phi); y = R*np.sin(phi)
vx = vR*np.cos(phi) - vt*np.sin(phi); vy = vR*np.sin(phi) + vt*np.cos(phi)
fin = ok & np.isfinite(x) & np.isfinite(vx) & np.isfinite(vz)

eos = halo & (feh > -0.9) & (feh < -0.2) & (mg > acc(feh)) & (mg < hl(feh)) & (al > c.alfe_cut) & fin
eos_hi = eos & (mg > divline(feh))     # alpha-rich (upper)
eos_lo = eos & (mg <= divline(feh))    # alpha-poor (lower)
disc = thin_al & (np.asarray(cat['galvt'], float) > 150) & (feh > -0.7) & (feh < -0.3) & fin

# --- actions ---
def actions(sel):
    xv = np.column_stack([x[sel], y[sel], zz[sel], vx[sel], vy[sel], vz[sel]])
    J = af(xv)                     # columns: Jr, Jz, Jphi
    return J[:, 0], J[:, 1], J[:, 2]
Jr_hi, Jz_hi, Jp_hi = actions(eos_hi)
Jr_lo, Jz_lo, Jp_lo = actions(eos_lo)
Jr_d, Jz_d, Jp_d = actions(disc)

def q(a): return np.percentile(a, [16, 50, 84])
print(f'alpha-rich (upper) n={eos_hi.sum():3d}  Jr 16/50/84 = {q(Jr_hi).round(0)}  Jz med={np.median(Jz_hi):.0f}  Lz med={np.median(Jp_hi):.0f}')
print(f'alpha-poor (lower) n={eos_lo.sum():3d}  Jr 16/50/84 = {q(Jr_lo).round(0)}  Jz med={np.median(Jz_lo):.0f}  Lz med={np.median(Jp_lo):.0f}')
ks = ks_2samp(Jr_hi, Jr_lo); mw = mannwhitneyu(Jr_hi, Jr_lo)
print(f'J_R  alpha-rich vs alpha-poor:  KS p={ks.pvalue:.3g}   Mann-Whitney p={mw.pvalue:.3g}')
# bootstrap error on median J_R difference
rng = np.random.default_rng(0)
d = np.array([np.median(rng.choice(Jr_lo, Jr_lo.size)) - np.median(rng.choice(Jr_hi, Jr_hi.size)) for _ in range(5000)])
print(f'median J_R(lower) - median J_R(upper) = {np.median(Jr_lo)-np.median(Jr_hi):+.0f} +- {d.std():.0f} kpc km/s')

# --- figure ---
CU, CL = '#e07a1f', '#2b6cb0'
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
# (1) J_R distribution
xg = np.linspace(0, 2500, 300)
for Jr, col, lab in [(Jr_d, '0.6', f'low-$\\alpha$ disc (ref, n={disc.sum()})'),
                     (Jr_hi, CU, f'Eos $\\alpha$-rich (n={eos_hi.sum()})'),
                     (Jr_lo, CL, f'Eos $\\alpha$-poor (n={eos_lo.sum()})')]:
    ax[0].hist(Jr, bins=np.linspace(0, 2500, 34), density=True, color=col, alpha=0.25)
    ax[0].plot(xg, gaussian_kde(Jr)(xg), color=col, lw=2.4,
               label=lab + f', med={np.median(Jr):.0f}')
    ax[0].axvline(np.median(Jr), color=col, ls=':', lw=1.4)
ax[0].set_xlim(0, 2500)
label_axes(ax[0], r'$J_R$ [kpc km/s]', 'density',
           f'Radial action of the two Eos branches (KS p={ks.pvalue:.2g})')
ax[0].legend(frameon=False, fontsize=9, loc='upper right')
# (2) action space: J_R vs L_z (Jphi)
ax[1].scatter(Jp_d, Jr_d, s=6, c='0.75', linewidths=0, zorder=0)
ax[1].scatter(Jp_hi, Jr_hi, s=22, c=CU, edgecolors='k', linewidths=0.3, zorder=3, label='$\\alpha$-rich')
ax[1].scatter(Jp_lo, Jr_lo, s=22, c=CL, edgecolors='k', linewidths=0.3, zorder=3, label='$\\alpha$-poor')
ax[1].axvline(0, color='k', lw=0.7, ls='--')
ax[1].set_xlim(-1200, 1600); ax[1].set_ylim(0, 2500)
label_axes(ax[1], r'$L_z\ (J_\phi)$ [kpc km/s]', r'$J_R$ [kpc km/s]', 'Action space: $J_R$ vs $L_z$')
ax[1].legend(frameon=False, fontsize=9, loc='upper right')
# (3) J_R vs J_z
ax[2].scatter(Jz_d, Jr_d, s=6, c='0.75', linewidths=0, zorder=0)
ax[2].scatter(Jz_hi, Jr_hi, s=22, c=CU, edgecolors='k', linewidths=0.3, zorder=3, label='$\\alpha$-rich')
ax[2].scatter(Jz_lo, Jr_lo, s=22, c=CL, edgecolors='k', linewidths=0.3, zorder=3, label='$\\alpha$-poor')
ax[2].set_xlim(0, 900); ax[2].set_ylim(0, 2500)
label_axes(ax[2], r'$J_z$ [kpc km/s]', r'$J_R$ [kpc km/s]', 'Vertical vs radial action')
ax[2].legend(frameon=False, fontsize=9, loc='upper right')
fig.suptitle('Eos actions (AGAMA, McMillan 2017 potential): is the radial action of the two branches different?', fontsize=12)
fig.savefig(FIG / '01_eos_actions.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_eos_actions.png')
