"""Test the GC-escapee hypothesis for the Eos high-N wing. Select N-rich Eos stars
(>P90 in [N/Fe] within each metallicity bin) and overplot them against the rest of
Eos in the GC-diagnostic planes: C-N, Na-O, Mg-Al. Second-generation GC stars are
N-rich, C-poor, Na-rich, O-poor, Al-rich, Mg-poor. NB: C-N anticorrelation ALSO comes
from first dredge-up in giants -> Mg-Al & Na-O are the clean (dredge-up-immune) tests.
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
cat = load_catalog('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_apogee_allspecies.fits.gz')
m = make_masks(cat, c)
G = {k: np.asarray(cat[k], float) for k in ['fe_h', 'galvt', 'n_fe', 'c_fe', 'o_fe', 'na_fe', 'mg_fe', 'al_fe']}
feh, vphi, nfe = G['fe_h'], G['galvt'], G['n_fe']
eos = np.asarray(m['thin_al'], bool) & (vphi > -75) & (vphi < 75)
disc = np.asarray(m['thin_al'], bool) & (vphi > 150) & (vphi < 300)
BINS = [(-0.8, -0.7), (-0.7, -0.6), (-0.6, -0.5)]
ROWS = [('c_fe', 'n_fe', '[C/Fe]', '[N/Fe]'),
        ('o_fe', 'na_fe', '[O/Fe]', '[Na/Fe]'),
        ('mg_fe', 'al_fe', '[Mg/Fe]', '[Al/Fe]')]
RNG = {'c_fe': (-0.55, 0.35), 'n_fe': (-0.3, 0.9), 'o_fe': (-0.15, 0.55),
       'na_fe': (-0.5, 0.9), 'mg_fe': (-0.05, 0.45), 'al_fe': (-0.5, 0.6)}

fig, ax = plt.subplots(3, 3, figsize=(13.5, 12), constrained_layout=True)
summary = {}
for j, (lo, hi) in enumerate(BINS):
    inbin = (feh >= lo) & (feh < hi)
    eb = eos & inbin & np.isfinite(nfe)
    p90 = np.percentile(nfe[eb], 90)
    nrich = eb & (nfe >= p90); rest = eb & (nfe < p90)
    summary[(lo, hi)] = (int(nrich.sum()), int(rest.sum()), p90)
    for i, (yc, xc, yl, xl) in enumerate(ROWS):
        a = ax[i, j]
        db = disc & inbin & np.isfinite(G[xc]) & np.isfinite(G[yc])
        a.scatter(G[xc][db], G[yc][db], s=5, c='0.8', rasterized=True, linewidths=0, label='low-$\\alpha$ disc' if i == 0 else None, zorder=0)
        rr = rest & np.isfinite(G[xc]) & np.isfinite(G[yc])
        nr = nrich & np.isfinite(G[xc]) & np.isfinite(G[yc])
        a.scatter(G[xc][rr], G[yc][rr], s=22, c='royalblue', edgecolors='k', linewidths=0.3, label='Eos (rest)', zorder=2)
        a.scatter(G[xc][nr], G[yc][nr], s=90, c='red', marker='*', edgecolors='k', linewidths=0.4, label='Eos N-rich (>P90)', zorder=3)
        a.set_xlim(*RNG[xc]); a.set_ylim(*RNG[yc])
        ttl = f'${lo}<$[Fe/H]$<{hi}$' if i == 0 else None
        label_axes(a, xl, yl, ttl)
        if i == 0 and j == 2:
            a.legend(frameon=False, fontsize=8, loc='upper right')
fig.suptitle('Is the Eos high-N wing GC 2G? — N-rich (red stars) vs rest of Eos in GC-diagnostic planes\n'
             '(GC 2G expects: C-poor, Na-rich/O-poor, Al-rich/Mg-poor;  Mg-Al & Na-O are dredge-up-immune)', fontsize=11)
fig.savefig(FIG / '01_eos_gc_signatures.png', dpi=140, bbox_inches='tight')

print('N-rich (>P90 [N/Fe]) vs rest of Eos, median [X/Fe] per bin:')
for (lo, hi), (nn, nr_, p90) in summary.items():
    inbin = (feh >= lo) & (feh < hi); eb = eos & inbin & np.isfinite(nfe)
    hi_m = eb & (nfe >= p90); lo_m = eb & (nfe < p90)
    print(f'--- {lo}<[Fe/H]<{hi}  N-rich n={nn}, rest n={nr_}, P90(N)={p90:+.2f} ---')
    for el in ['c_fe', 'o_fe', 'na_fe', 'mg_fe', 'al_fe']:
        y = G[el]
        mh = np.nanmedian(y[hi_m]); ml = np.nanmedian(y[lo_m])
        print(f'    {el:6s}  N-rich={mh:+.3f}  rest={ml:+.3f}  diff={mh-ml:+.3f}')
print('wrote', FIG / '01_eos_gc_signatures.png')
