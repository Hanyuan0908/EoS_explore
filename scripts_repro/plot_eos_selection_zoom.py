"""Zoom into the Eos selection to diagnose why the current cut misses low-alpha Eos stars.
LEFT  = CURRENT cut  (thin_al & Vtan<80 & -0.9<[Fe/H]<-0.5)  -- clips the metal-rich side.
RIGHT = REVISED cut  (Davies halo [ecc>0.7|Lz<0] & thin_al & -1.0<[Fe/H]<-0.3) -- consistent
        with the Davies-halo background and captures the full low-alpha overdensity.
Both over the Davies-halo [Mg/Fe]-[Fe/H] density; stars classified by the Davies divider.
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
lz = np.asarray(cat['lz'], float); rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float)
ecc = (rap - rperi) / (rap + rperi)
base = np.asarray(m['base'], bool); thin_al = np.asarray(m['thin_al'], bool)
halo = base & ((ecc > 0.7) | (lz < 0))
def divline(f): return 0.317*f + 0.353
def acc(f): return c.slope_acc*f + c.inter_acc
def hl(f): return c.slope_acc2*f + c.inter_acc2
CHI, CLO = 'magenta', 'cyan'
XR, YR = (-1.2, -0.1), (0.0, 0.33)

def draw_bg(ax):
    s = halo & np.isfinite(feh) & np.isfinite(mg)
    h, xe, ye = np.histogram2d(feh[s], mg[s], bins=[90, 60], range=[XR, YR])
    him = np.full_like(h, np.nan); him[h > 0] = np.log10(h[h > 0])
    ax.imshow(him.T, origin='lower', extent=[*XR, *YR], aspect='auto', cmap='Greys',
              vmin=np.nanpercentile(him, 2), vmax=np.nanpercentile(him, 99), zorder=0)
    xx = np.linspace(*XR, 60)
    ax.plot(xx, hl(xx), 'r:', lw=1.4, zorder=2, label='high/low-$\\alpha$ line')
    ax.plot(xx, acc(xx), 'r--', lw=1.1, zorder=2, label='accreted line')
    ax.plot(xx, divline(xx), 'g--', lw=1.6, zorder=2, label='Davies divider')
    ax.set_xlim(*XR); ax.set_ylim(*YR)

def classify_plot(ax, sel):
    hi = sel & (mg > divline(feh)); lo = sel & (mg <= divline(feh))
    ax.scatter(feh[hi], mg[hi], s=16, c=CHI, edgecolors='k', linewidths=0.3, zorder=5,
               label=f'$\\alpha$-rich (n={int(hi.sum())})')
    ax.scatter(feh[lo], mg[lo], s=16, c=CLO, edgecolors='k', linewidths=0.3, zorder=5,
               label=f'$\\alpha$-poor (n={int(lo.sum())})')
    return hi, lo

fig, ax = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
# --- current ---
draw_bg(ax[0])
cur = thin_al & (vphi < 80) & (feh > -0.9) & (feh < -0.5)
for xv in (-0.9, -0.5): ax[0].axvline(xv, color='navy', ls='-', lw=1.2, zorder=3)
classify_plot(ax[0], cur)
label_axes(ax[0], '[Fe/H]', '[Mg/Fe]', f'CURRENT: thin_al & $V_{{tan}}<80$ & $-0.9<$[Fe/H]$<-0.5$ (n={int(cur.sum())})')
ax[0].legend(frameon=False, fontsize=8.5, loc='upper right')
# --- revised ---
draw_bg(ax[1])
FLO, FHI = -1.0, -0.3
rev = halo & thin_al & (feh > FLO) & (feh < FHI)
for xv in (FLO, FHI): ax[1].axvline(xv, color='navy', ls='-', lw=1.2, zorder=3)
classify_plot(ax[1], rev)
label_axes(ax[1], '[Fe/H]', '[Mg/Fe]', f'REVISED: Davies halo & thin_al & ${FLO}<$[Fe/H]$<{FHI}$ (n={int(rev.sum())})')
ax[1].legend(frameon=False, fontsize=8.5, loc='upper right')
fig.suptitle('Eos selection: the current box clips the metal-rich low-$\\alpha$ overdensity; revised cut captures it', fontsize=12)
fig.savefig(FIG / '01_eos_selection_zoom.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_eos_selection_zoom.png')
print(f'CURRENT n={int(cur.sum())}; REVISED n={int(rev.sum())}')
