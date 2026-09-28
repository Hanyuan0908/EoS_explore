"""[McMillan17 series] E-Lz for accreted / high-a / low-a, from the from-scratch
BJ21+DR3 -> AGAMA+McMillan17 kinematics (scripts_repro/eos_mcm.py). Same reference
plotting helpers as Fig_paper/obs_energy_pops, but energy uses the McMillan17
zero-point so the E axis range is retuned. Output Fig_paper_mcmillan17/obs_energy_pops.{pdf,png}
"""
import os, sys
import numpy as np
REPO = '/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore'
sys.path.insert(0, REPO + '/scripts_repro'); sys.path.insert(0, REPO + '/eos-figures')
from eos_figures.plotting import setup_axes, density_panel, label_axes
from eos_figures.stats import hist2d
from eos_figures.config import Cuts
from eos_mcm import load_mcm, make_masks_mcm
OUT = REPO + '/Fig_paper_mcmillan17'; os.makedirs(OUT, exist_ok=True)
c = Cuts(); cat = load_mcm(); m = make_masks_mcm(cat, c)
lz = np.asarray(cat['lz'], float); en = np.asarray(cat['energy'], float)
ENR = (-2.35, -0.85)                       # E*1e-5, McMillan17 zero-point
fig, ax = setup_axes(3, figsize=(10, 3))
for axis, mask_name, title in zip(ax, ['acc_al', 'thick_al', 'thin_al'],
                                  ['accreted', r'high-$\alpha$', r'low-$\alpha$']):
    mm = np.asarray(m[mask_name], bool)
    h, xe, ye = hist2d(lz[mm], 1e-5 * en[mm], c.lzr, ENR, c.nlz2, c.nen2)
    im = density_panel(axis, h, xe * 1e-3, ye, percentiles=c.perc_elz, vmin=-0.3)
    im.set_rasterized(True)
    axis.axvline(0, color='k', lw=0.8)
    label_axes(axis, r'$L_z\times 10^{-3}$', r'$E\times 10^{-5}$', title)
ax[2].text(-1.35, -2.05, 'Eos', fontsize=9)
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/obs_energy_pops.{ext}', bbox_inches='tight')
print('wrote', OUT + '/obs_energy_pops.{pdf,png}')
