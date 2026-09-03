"""Clean distributions of J_R (left) and J_R/|Lz| (right) for the two Eos branches
(alpha-rich upper / alpha-poor lower), canonical cut. Actions via AGAMA (McMillan 2017).
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, ks_2samp
from astropy.io import fits
import agama
REPO = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/eos-figures')
sys.path.insert(0, str(REPO))
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
from eos_figures.plotting import label_axes
c = Cuts()
FIG = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/figures_repro')
agama.setUnits(mass=1, length=1, velocity=1)
pot = agama.Potential(os.path.join(os.path.dirname(agama.__file__), 'data', 'McMillan17.ini'))
af = agama.ActionFinder(pot)

cat = load_catalog('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c); base = np.asarray(m['base'], bool)
feh = np.asarray(cat['fe_h'], float); mg = np.asarray(cat['mg_fe'], float); al = np.asarray(cat['al_fe'], float)
lz = np.asarray(cat['lz'], float); rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float)
ecc = (rap - rperi)/(rap + rperi); aid = np.asarray(cat['apogee_id'])
halo = base & ((ecc > 0.7) | (lz < 0))
ann = fits.open('/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE/apogee_astroNN-DR17.fits')[1].data
def norm(a): return np.array([(s.decode() if isinstance(s, bytes) else str(s)).strip() for s in np.asarray(a)])
nid = norm(ann['APOGEE_ID']); o = np.argsort(nid); nid_s = nid[o]
p = np.clip(np.searchsorted(nid_s, aid), 0, len(nid_s)-1); ok = nid_s[p] == aid; src = o[p]
cc = lambda n: np.where(ok, np.asarray(ann[n], float)[src], np.nan)
R, phi, zz = cc('galr'), cc('galphi'), cc('galz'); vR, vt, vz = cc('galvr'), cc('galvt'), cc('galvz')
x = R*np.cos(phi); y = R*np.sin(phi); vx = vR*np.cos(phi)-vt*np.sin(phi); vy = vR*np.sin(phi)+vt*np.cos(phi)
fin = ok & np.isfinite(x) & np.isfinite(vx) & np.isfinite(vz)

def acc(f): return c.slope_acc*f + c.inter_acc
def hl(f): return c.slope_acc2*f + c.inter_acc2
def divline(f): return 0.317*f + 0.353
eos = halo & (feh > -0.9) & (feh < -0.2) & (mg > acc(feh)) & (mg < hl(feh)) & (al > c.alfe_cut) & fin
eos_hi = eos & (mg > divline(feh)); eos_lo = eos & (mg <= divline(feh))

def actions(s):
    J = af(np.column_stack([x[s], y[s], zz[s], vx[s], vy[s], vz[s]]))
    return J[:, 0], J[:, 2]
Jr_hi, Lz_hi = actions(eos_hi); Jr_lo, Lz_lo = actions(eos_lo)
rat_hi = Jr_hi/np.maximum(np.abs(Lz_hi), 1.0); rat_lo = Jr_lo/np.maximum(np.abs(Lz_lo), 1.0)
CU, CL = '#e07a1f', '#2b6cb0'

fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.0), constrained_layout=True)
# --- (1) J_R distribution ---
xg = np.linspace(0, 2000, 300)
ksR = ks_2samp(Jr_hi, Jr_lo).pvalue
for Jr, col, lab in [(Jr_hi, CU, r'Eos $\alpha$-rich'), (Jr_lo, CL, r'Eos $\alpha$-poor')]:
    ax[0].hist(Jr, bins=np.linspace(0, 2000, 26), density=True, color=col, alpha=0.22)
    ax[0].plot(xg, gaussian_kde(Jr)(xg), color=col, lw=2.6, label=f'{lab} (n={Jr.size}, med={np.median(Jr):.0f})')
    ax[0].axvline(np.median(Jr), color=col, ls=':', lw=1.6)
ax[0].set_xlim(0, 2000)
label_axes(ax[0], r'$J_R$ [kpc km/s]', 'density', f'Radial action of the two Eos branches (KS $p$={ksR:.1g})')
ax[0].legend(frameon=False, fontsize=10, loc='upper right')
# --- (2) J_R/|Lz| distribution (log x for the heavy tail) ---
lg_hi, lg_lo = np.log10(rat_hi), np.log10(rat_lo)
xl = np.linspace(-1.3, 2.3, 300)
ksr = ks_2samp(rat_hi, rat_lo).pvalue
for lg, rat, col, lab in [(lg_hi, rat_hi, CU, r'Eos $\alpha$-rich'), (lg_lo, rat_lo, CL, r'Eos $\alpha$-poor')]:
    ax[1].hist(lg, bins=np.linspace(-1.3, 2.3, 26), density=True, color=col, alpha=0.22)
    ax[1].plot(xl, gaussian_kde(lg)(xl), color=col, lw=2.6, label=f'{lab} (med={np.median(rat):.1f})')
    ax[1].axvline(np.log10(np.median(rat)), color=col, ls=':', lw=1.6)
ax[1].set_xlim(-1.3, 2.3)
ticks = [0.1, 0.3, 1, 3, 10, 30, 100]
ax[1].set_xticks(np.log10(ticks)); ax[1].set_xticklabels([str(t) for t in ticks])
label_axes(ax[1], r'$J_R/|L_z|$', 'density', f'Radial anisotropy ratio (KS $p$={ksr:.1g})')
ax[1].legend(frameon=False, fontsize=10, loc='upper right')
fig.suptitle('Eos branches: radial action and $J_R/|L_z|$ distributions (AGAMA / McMillan 2017)', fontsize=12)
fig.savefig(FIG / '01_eos_action_dists.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_eos_action_dists.png')
print(f'J_R    : a-rich med={np.median(Jr_hi):.0f}  a-poor med={np.median(Jr_lo):.0f}  KS p={ksR:.2g}')
print(f'J_R/|Lz|: a-rich med={np.median(rat_hi):.2f}  a-poor med={np.median(rat_lo):.2f}  KS p={ksr:.2g}')
