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
feh = np.asarray(cat['fe_h'], float); vphi = np.asarray(cat['galvt'], float); nfe = np.asarray(cat['n_fe'], float)
# N_FE_FLAG (to mark GRIDEDGE_WARN grid-edge stars) from allStar, matched by APOGEE_ID (dedup best SNR)
from astropy.io import fits
_d = fits.open('/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE/APOGEE_DR17_all.fits')[1].data
def _norm(a): return np.array([(s.decode() if isinstance(s, bytes) else str(s)).strip() for s in np.asarray(a)])
_id = _norm(_d['APOGEE_ID']); _o = np.argsort(-np.asarray(_d['SNR'], float))
_ids = _id[_o]; _fls = np.asarray(_d['N_FE_FLAG'], np.int64)[_o]
_, _fi = np.unique(_ids, return_index=True); _uid = _ids[_fi]; _ufl = _fls[_fi]
_sr = np.argsort(_uid); _uid = _uid[_sr]; _ufl = _ufl[_sr]
_aid = np.asarray(cat['apogee_id']); _pos = np.clip(np.searchsorted(_uid, _aid), 0, len(_uid) - 1)
gridedge = np.where(_uid[_pos] == _aid, _ufl[_pos], 1) != 0   # True = N_FE_FLAG!=0 (grid-edge)
eos = np.asarray(m['thin_al'], bool) & (vphi > -75) & (vphi < 75)      # Eos block
disc = np.asarray(m['thin_al'], bool) & (vphi > 150) & (vphi < 300)    # low-a disc (reference)
BINS = [(-0.8, -0.7), (-0.7, -0.6), (-0.6, -0.5)]

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True, constrained_layout=True)
xg = np.linspace(-0.45, 0.75, 240); hb = np.linspace(-0.45, 0.75, 26)
for a, (lo, hi) in zip(ax, BINS):
    inbin = (feh >= lo) & (feh < hi) & np.isfinite(nfe)
    ecl = eos & inbin & ~gridedge; egr = eos & inbin & gridedge     # clean vs grid-edge
    yd = nfe[disc & inbin]
    # stacked histogram: clean (blue) + grid-edge (crimson)
    a.hist([nfe[ecl], nfe[egr]], bins=hb, density=True, stacked=True,
           color=['royalblue', 'crimson'], alpha=0.75,
           label=[f'Eos clean (n={int(ecl.sum())})', f'Eos GRIDEDGE_WARN (n={int(egr.sum())})'])
    a.plot(xg, gaussian_kde(nfe[ecl])(xg), color='royalblue', lw=1.6)
    a.plot(xg, gaussian_kde(yd)(xg), color='0.35', ls='--', lw=1.6, label=f'low-$\\alpha$ disc (n={yd.size})')
    ye = nfe[eos & inbin]
    p5, p50, p95 = np.percentile(ye, [5, 50, 95])
    for v, lab in [(p5, 'P5'), (p95, 'P95')]:
        a.axvline(v, color='navy', ls=':', lw=1.1)
        a.text(v, a.get_ylim()[1] * 0.02, lab, fontsize=7, color='navy', rotation=90, va='bottom', ha='right')
    a.axvline(np.median(yd), color='0.35', ls=':', lw=1.0)
    a.set_xlim(-0.45, 0.75)
    a.text(0.97, 0.7, f'Eos P95$-$P5 = {p95 - p5:.2f}\ndisc = {np.percentile(yd,95)-np.percentile(yd,5):.2f}',
           transform=a.transAxes, ha='right', fontsize=8, color='0.25')
    label_axes(a, '[N/Fe]', 'density' if a is ax[0] else '', f'${lo}<$[Fe/H]$<{hi}$')
    a.legend(frameon=False, fontsize=7.5, loc='upper left')
fig.suptitle('Eos [N/Fe] distribution per metallicity bin — grid-edge (crimson) makes the low-N tail; the real high-N wing is clean', fontsize=11)
fig.savefig(FIG / '01_eos_Ndist_bins.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_eos_Ndist_bins.png')
for lo, hi in BINS:
    ye = nfe[eos & (feh >= lo) & (feh < hi) & np.isfinite(nfe)]
    print(f'  {lo}<feh<{hi}: n={ye.size}  med={np.median(ye):+.3f}  P5={np.percentile(ye,5):+.3f}  P95={np.percentile(ye,95):+.3f}')
