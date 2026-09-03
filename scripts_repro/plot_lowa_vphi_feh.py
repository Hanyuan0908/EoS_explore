"""Clean V_tan (~V_phi) vs [Fe/H] plane of the low-alpha (in-situ, thin_al) population.
No selections, no overlays -- just the 2D density of the low-alpha stars.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
REPO = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/eos-figures')
sys.path.insert(0, str(REPO))
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
c = Cuts()
FIG = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/figures_repro')
cat = load_catalog('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c)
feh = np.asarray(cat['fe_h'], float); vphi = np.asarray(cat['galvt'], float)
lowa = np.asarray(m['thin_al'], bool) & np.isfinite(feh) & np.isfinite(vphi)

fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
h = ax.hist2d(feh[lowa], vphi[lowa], bins=[140, 140],
              range=[(-1.2, 0.55), (-250, 400)], cmap='viridis', norm=LogNorm())
fig.colorbar(h[3], ax=ax, pad=0.01).set_label('count')
ax.set_xlabel('[Fe/H] [dex]')
ax.set_ylabel(r'$V_{\rm tan}\ (\approx V_\phi)$ [km/s]')
ax.set_title(rf'Low-$\alpha$ (in-situ) population: $V_\phi$ vs [Fe/H]  (n={int(lowa.sum())})')
fig.savefig(FIG / '01_lowa_vphi_feh.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_lowa_vphi_feh.png', 'n=', int(lowa.sum()))
