"""V_R vs V_phi (velocity plane) of the two Eos branches.
V_phi is in the lite cache (galvt); V_R must be matched from the AstroNN VAC by APOGEE_ID.
Eos = low-alpha (thin_al) & -0.9<[Fe/H]<-0.5, halo-selected by ecc>0.7 (no vphi bias),
split into alpha-rich (upper) / alpha-poor (lower) by the Davies divider mg=0.317*feh+0.353.
Low-alpha population shown as a grey density background.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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
feh = np.asarray(cat['fe_h'], float); mg = np.asarray(cat['mg_fe'], float)
al = np.asarray(cat['al_fe'], float); lz = np.asarray(cat['lz'], float)
vphi = np.asarray(cat['galvt'], float)
rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float)
ecc = (rap - rperi) / (rap + rperi); aid = np.asarray(cat['apogee_id'])
base = np.asarray(m['base'], bool); thin_al = np.asarray(m['thin_al'], bool)

# match V_R from AstroNN VAC by APOGEE_ID
ann = fits.open('/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE/apogee_astroNN-DR17.fits')[1].data
def norm(a): return np.array([(s.decode() if isinstance(s, bytes) else str(s)).strip() for s in np.asarray(a)])
nid = norm(ann['APOGEE_ID']); o = np.argsort(nid); nid_s = nid[o]
p = np.clip(np.searchsorted(nid_s, aid), 0, len(nid_s)-1); ok = nid_s[p] == aid; src = o[p]
vR = np.where(ok, np.asarray(ann['galvr'], float)[src], np.nan)
fin = ok & np.isfinite(vR) & np.isfinite(vphi)

# CANONICAL Eos cut: Davies halo & low-alpha wedge & -0.9<[Fe/H]<-0.2
halo = base & ((ecc > 0.7) | (lz < 0))
def acc(f): return c.slope_acc*f + c.inter_acc
def hl(f): return c.slope_acc2*f + c.inter_acc2
def divline(f): return 0.317 * f + 0.353
eos = halo & (feh > -0.9) & (feh < -0.2) & (mg > acc(feh)) & (mg < hl(feh)) & (al > c.alfe_cut) & fin
eos_hi = eos & (mg > divline(feh))     # alpha-rich (upper)
eos_lo = eos & (mg <= divline(feh))    # alpha-poor (lower)
CU, CL = '#e07a1f', '#2b6cb0'

fig, ax = plt.subplots(figsize=(7.8, 7.2), constrained_layout=True)
# grey low-alpha background density
bg = thin_al & fin
ax.hist2d(vphi[bg], vR[bg], bins=[120, 120], range=[(-250, 400), (-250, 250)],
          cmap='Greys', norm=LogNorm(), zorder=0)
ax.scatter(vphi[eos_hi], vR[eos_hi], s=26, c=CU, edgecolors='k', linewidths=0.3,
           label=fr'Eos $\alpha$-rich (upper), n={int(eos_hi.sum())}', zorder=3)
ax.scatter(vphi[eos_lo], vR[eos_lo], s=26, c=CL, edgecolors='k', linewidths=0.3,
           label=fr'Eos $\alpha$-poor (lower), n={int(eos_lo.sum())}', zorder=3)
ax.axhline(0, color='0.4', lw=0.8, ls='--'); ax.axvline(0, color='0.4', lw=0.8, ls='--')
# mean markers
for sel, col in [(eos_hi, CU), (eos_lo, CL)]:
    ax.scatter(vphi[sel].mean(), vR[sel].mean(), s=320, marker='*',
               c=col, edgecolors='k', linewidths=1.3, zorder=5)
ax.set_xlim(-250, 400); ax.set_ylim(-250, 250)
label_axes(ax, r'$V_\phi\ (\equiv V_{\rm tan})$ [km/s]', r'$V_R$ [km/s]',
           r'Velocity plane of the two Eos branches (bg = low-$\alpha$ density; $\star$ = branch mean)')
ax.legend(frameon=False, fontsize=9.5, loc='upper left')
fig.savefig(FIG / '01_eos_vr_vphi.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_eos_vr_vphi.png')
for nm, s in [('alpha-rich', eos_hi), ('alpha-poor', eos_lo), ('both', eos)]:
    n = int(s.sum())
    print(f'  {nm:11s} n={n:4d}  <Vphi>={vphi[s].mean():+6.1f}+-{vphi[s].std()/np.sqrt(n):4.1f} '
          f'(disp {vphi[s].std():5.1f})  <VR>={vR[s].mean():+6.1f}+-{vR[s].std()/np.sqrt(n):4.1f} '
          f'(disp {vR[s].std():5.1f})')
