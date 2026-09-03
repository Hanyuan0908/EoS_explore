"""Is the Eos Mg 'bifurcation' actually two separable populations, or one continuous
sequence? LEFT: distribution of dMg = [Mg/Fe] - divider (Davies line) for the kinematic
Eos sample -> test bimodality. MIDDLE: [Mg/Fe]-[Fe/H] scatter with a BUFFER GAP of +-GAP
dex around the divider (ambiguous middle dropped) to isolate pure alpha-rich / alpha-poor.
RIGHT: AstroNN age distributions of the gap-separated pure ends vs same-[Fe/H] disc.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
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
al = np.asarray(cat['al_fe'], float)
age = np.asarray(cat['age'], float); aerr = np.asarray(cat['age_model_error'], float)
base = np.asarray(m['base'], bool); thin_al = np.asarray(m['thin_al'], bool)
lz = np.asarray(cat['lz'], float); rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float)
ecc = (rap - rperi) / (rap + rperi)
halo = base & ((ecc > 0.7) | (lz < 0))          # Davies halo sample
rel_ok = np.isfinite(age) & np.isfinite(aerr) & (aerr/age < 0.3)

def acc(f): return c.slope_acc*f + c.inter_acc
def hl(f): return c.slope_acc2*f + c.inter_acc2
def divline(f): return 0.317*f + 0.353
GAP = 0.03                                # +-dex buffer around the divider
# CANONICAL Eos cut: Davies halo & low-alpha wedge & -0.9<[Fe/H]<-0.2
eos = halo & (feh > -0.9) & (feh < -0.2) & (mg > acc(feh)) & (mg < hl(feh)) & (al > c.alfe_cut)
dmg = mg - divline(feh)
eos_hi = eos & (dmg > +GAP)               # pure alpha-rich
eos_lo = eos & (dmg < -GAP)               # pure alpha-poor
eos_mid = eos & (np.abs(dmg) <= GAP)      # ambiguous (dropped)
disc = thin_al & (vphi > 150)
CHI, CLO = 'magenta', 'cyan'

fig, ax = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw={'width_ratios': [1, 1, 1.3]}, constrained_layout=True)
# --- (1) Davies halo [Mg/Fe]-[Fe/H] density + Eos branch scatters ---
sh = halo & np.isfinite(feh) & np.isfinite(mg)
hh, xh, yh = np.histogram2d(feh[sh], mg[sh], bins=[70, 55], range=[(-2.1, 0.6), (-0.1, 0.5)])
hhim = np.full_like(hh, np.nan); hhim[hh > 0] = np.log10(hh[hh > 0])
ax[0].imshow(hhim.T, origin='lower', extent=[-2.1, 0.6, -0.1, 0.5], aspect='auto', cmap='Greys',
             vmin=np.nanpercentile(hhim, 2), vmax=np.nanpercentile(hhim, 99), zorder=0)
ax[0].scatter(feh[eos_mid], mg[eos_mid], s=10, c='0.6', linewidths=0, zorder=3)
ax[0].scatter(feh[eos_hi], mg[eos_hi], s=16, c=CHI, edgecolors='k', linewidths=0.3, zorder=5)
ax[0].scatter(feh[eos_lo], mg[eos_lo], s=16, c=CLO, edgecolors='k', linewidths=0.3, zorder=5)
ax[0].set_xlim(-2.1, 0.6); ax[0].set_ylim(-0.1, 0.5)
label_axes(ax[0], '[Fe/H]', '[Mg/Fe]', f'Davies halo ($e>0.7|L_z<0$), n={int(sh.sum())}')
# --- (2) scatter with the gap applied ---
s = base & np.isfinite(feh) & np.isfinite(mg)
h, xe, ye = np.histogram2d(feh[s], mg[s], bins=[80, 60], range=[(-1.5, 0.5), (-0.05, 0.45)])
him = np.full_like(h, np.nan); him[h > 0] = np.log10(h[h > 0])
ax[1].imshow(him.T, origin='lower', extent=[-1.5, 0.5, -0.05, 0.45], aspect='auto', cmap='Greys',
             vmin=np.nanpercentile(him, 3), vmax=np.nanpercentile(him, 99.5), zorder=0)
xx = np.linspace(-1.5, 0.5, 60)
ax[1].plot(xx, divline(xx), 'g--', lw=1.4, zorder=2)
ax[1].fill_between(xx, divline(xx)-GAP, divline(xx)+GAP, color='0.5', alpha=0.3, zorder=1)
ax[1].scatter(feh[eos_mid], mg[eos_mid], s=10, c='0.6', linewidths=0, zorder=3, label=f'dropped (n={int(eos_mid.sum())})')
ax[1].scatter(feh[eos_hi], mg[eos_hi], s=16, c=CHI, edgecolors='k', linewidths=0.3, zorder=5, label=f'$\\alpha$-rich (n={int(eos_hi.sum())})')
ax[1].scatter(feh[eos_lo], mg[eos_lo], s=16, c=CLO, edgecolors='k', linewidths=0.3, zorder=5, label=f'$\\alpha$-poor (n={int(eos_lo.sum())})')
ax[1].set_xlim(-1.0, -0.15); ax[1].set_ylim(0.02, 0.32)
label_axes(ax[1], '[Fe/H]', '[Mg/Fe]', f'Buffer gap $\\pm${GAP} isolates the pure ends')
ax[1].legend(frameon=False, fontsize=9, loc='upper right')
# --- (3) ages of the pure ends vs matched-metallicity disc ---
ag = np.linspace(0.5, 14, 300)
from matplotlib import cm, colors
cmap = cm.coolwarm; norm = colors.Normalize(-0.8, 0.4)
edges = np.arange(-0.8, -0.2 + 1e-9, 0.1)
ymax = 0
for i in range(len(edges)-1):
    b = disc & rel_ok & (feh >= edges[i]) & (feh < edges[i+1])
    if age[b].size >= 50: ymax = max(ymax, gaussian_kde(age[b])(ag).max())
for i in range(len(edges)-1):
    lo, hi = edges[i], edges[i+1]; fc = 0.5*(lo+hi)
    b = disc & rel_ok & (feh >= lo) & (feh < hi); y = age[b]
    if y.size >= 50:
        ax[2].plot(ag, gaussian_kde(y)(ag), color=cmap(norm(fc)), lw=3.5, alpha=0.9, zorder=3,
                   label=f'disc {lo:.1f}<[Fe/H]<{hi:.1f}')
for sel, col, lab in [(eos_hi, CHI, r'Eos $\alpha$-rich (pure)'), (eos_lo, CLO, r'Eos $\alpha$-poor (pure)')]:
    y = age[sel & rel_ok]
    ax[2].plot(ag, gaussian_kde(y)(ag), color=col, lw=3.5, zorder=6,
               label=f'{lab} (n={y.size}, med={np.median(y):.1f})')
ax[2].set_xlim(0.5, 14); ax[2].set_ylim(0, 1.1*ymax)
label_axes(ax[2], 'age [Gyr] (AstroNN)', 'number density', 'Ages of the gap-separated pure ends')
ax[2].legend(frameon=False, fontsize=8.5, loc='upper left')
fig.savefig(FIG / '01_eos_branch_gap.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_eos_branch_gap.png')
for sel, lab in [(eos_hi, 'a-rich pure'), (eos_lo, 'a-poor pure'), (eos_mid, 'dropped mid')]:
    y = age[sel & rel_ok]
    print(f'  {lab:12s} n_all={int(sel.sum()):3d} n_age={y.size:3d} med_age={np.median(y):.1f} '
          f'feh_med={np.median(feh[sel]):+.2f} mg_med={np.median(mg[sel]):.2f}')
