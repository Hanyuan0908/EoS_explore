"""[N/Fe] vs [Fe/H], COLUMN-NORMALISED density (each [Fe/H] column scaled to its own max),
with the P5 and P95 percentiles of [N/Fe] traced as a function of [Fe/H] (the two lines).
In-situ sample (thin_al | thick_al); [Fe/H] from -1.5 to +0.5.
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
feh = np.asarray(cat['fe_h'], float); nfe = np.asarray(cat['n_fe'], float)
insitu = (np.asarray(m['thin_al'], bool) | np.asarray(m['thick_al'], bool)) & np.isfinite(feh) & np.isfinite(nfe)

FEHR = (-1.5, 0.5); NR = (-0.45, 0.85)
NX, NY = 80, 70
H, xe, ye = np.histogram2d(feh[insitu], nfe[insitu], bins=[NX, NY], range=[FEHR, NR])
colmax = H.max(axis=1, keepdims=True)                       # column-normalise: each [Fe/H] column / its max
Hn = np.divide(H, colmax, out=np.full_like(H, np.nan), where=colmax > 0)

fig, ax = plt.subplots(figsize=(8.5, 5.6), constrained_layout=True)
ax.imshow(Hn.T, origin='lower', extent=[*FEHR, *NR], aspect='auto', cmap='Greys', vmin=0, vmax=1, zorder=0)

# P5 / P95 (the two lines) + median, per [Fe/H] column (>=25 stars)
edges = np.arange(FEHR[0], FEHR[1] + 1e-9, 0.05); cen = 0.5*(edges[:-1] + edges[1:])
p5 = np.full(len(cen), np.nan); p50 = np.full(len(cen), np.nan); p95 = np.full(len(cen), np.nan)
for i in range(len(cen)):
    y = nfe[insitu & (feh >= edges[i]) & (feh < edges[i+1])]
    if y.size >= 25:
        p5[i], p50[i], p95[i] = np.percentile(y, [5, 50, 95])
ax.plot(cen, p95, color='crimson', lw=2.4, label='[N/Fe] P95')
ax.plot(cen, p5,  color='royalblue', lw=2.4, label='[N/Fe] P5')
ax.plot(cen, p50, color='0.35', lw=1.4, ls='--', label='median')
ax.set_xlim(*FEHR); ax.set_ylim(*NR)
label_axes(ax, '[Fe/H]', '[N/Fe]', 'In-situ [N/Fe]$-$[Fe/H] (column-normalised) with P5 / P95 tracks')
ax.legend(frameon=False, fontsize=10, loc='upper right')
fig.savefig(FIG / '01_nfe_feh_pctl.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_nfe_feh_pctl.png')
for lbl, arr in [('P5', p5), ('P50', p50), ('P95', p95)]:
    print(lbl, 'at [Fe/H]=-1.4,-1.0,-0.6,-0.2:', [round(np.interp(v, cen, arr), 2) for v in (-1.4, -1.0, -0.6, -0.2)])
