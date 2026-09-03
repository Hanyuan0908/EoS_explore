"""V_tan-[Fe/H] plane with the low-alpha population as background density and the two Eos
branches (alpha-rich upper vs alpha-poor lower, Davies divider) over-plotted, colour-coded.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
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
thin_al = np.asarray(m['thin_al'], bool)
# CANONICAL Eos cut: Davies halo & low-alpha wedge & -0.9<[Fe/H]<-0.2
al = np.asarray(cat['al_fe'], float); lz = np.asarray(cat['lz'], float)
rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float); ecc = (rap - rperi)/(rap + rperi)
base = np.asarray(m['base'], bool)
halo = base & ((ecc > 0.7) | (lz < 0))
def acc(f): return c.slope_acc*f + c.inter_acc
def hl(f): return c.slope_acc2*f + c.inter_acc2
def divline(f): return 0.317*f + 0.353
eos = halo & (feh > -0.9) & (feh < -0.2) & (mg > acc(feh)) & (mg < hl(feh)) & (al > c.alfe_cut)
eos_hi = eos & (mg > divline(feh)); eos_lo = eos & (mg <= divline(feh))
CHI, CLO = 'magenta', 'cyan'
FEHR = (-1.5, 0.5); VR = (-200, 350)

fig, ax = plt.subplots(figsize=(8, 5.6), constrained_layout=True)
# background: low-alpha population density
s = thin_al & np.isfinite(feh) & np.isfinite(vphi)
h, xe, ye = np.histogram2d(feh[s], vphi[s], bins=[90, 70], range=[FEHR, VR])
him = np.full_like(h, np.nan); him[h > 0] = np.log10(h[h > 0])
ax.imshow(him.T, origin='lower', extent=[*FEHR, *VR], aspect='auto', cmap='Greys',
          vmin=np.nanpercentile(him, 3), vmax=np.nanpercentile(him, 99.5), zorder=0)
ax.axhline(0, color='k', ls='--', lw=0.8, zorder=1)
ax.axhline(80, color='0.5', ls=':', lw=1.0, zorder=1)
ax.scatter(feh[eos_hi], vphi[eos_hi], s=26, c=CHI, edgecolors='k', linewidths=0.4, zorder=4,
           label=f'Eos $\\alpha$-rich (upper, n={int(eos_hi.sum())})')
ax.scatter(feh[eos_lo], vphi[eos_lo], s=26, c=CLO, edgecolors='k', linewidths=0.4, zorder=4,
           label=f'Eos $\\alpha$-poor (lower, n={int(eos_lo.sum())})')
ax.set_xlim(*FEHR); ax.set_ylim(*VR)
label_axes(ax, '[Fe/H]', r'$V_{\rm tan}$ [km/s]', r'Two Eos branches in $V_{\rm tan}$-[Fe/H] (bg = low-$\alpha$)')
ax.legend(frameon=False, fontsize=9, loc='upper left')
fig.savefig(FIG / '01_eos_vtan_branches.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_eos_vtan_branches.png')
for sel, lab in [(eos_hi, 'alpha-rich'), (eos_lo, 'alpha-poor')]:
    print(f'  Eos {lab:11s}: n={int(sel.sum())}  Vtan med={np.median(vphi[sel]):+.0f}  Vtan 16-84=[{np.percentile(vphi[sel],16):+.0f},{np.percentile(vphi[sel],84):+.0f}]  [Fe/H] med={np.median(feh[sel]):+.2f}')
