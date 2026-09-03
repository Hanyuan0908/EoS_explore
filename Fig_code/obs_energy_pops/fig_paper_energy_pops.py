"""Publication figure (observational): E-Lz for the three populations, reproduced
exactly from eos_figures.figures.plot_energy_pops (== figures_repro/01_fig3_energy_pops.png).

Three panels, accreted / high-a / low-a, log-density in the (Lz, E) plane with the
Lz = 0 line; the Eos overdensity is labelled in the low-a panel.

Data: data_repro/our_apogee_dr17_lite_ann.fits.gz (in-repo, portable).
Writes Fig_paper/obs_energy_pops.pdf and .png.
"""
import os
import sys
import numpy as np

REPO = '/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore'
sys.path.insert(0, REPO + '/eos-figures')
# importing eos_figures.plotting applies the reference repo rcParams -- do not override.
from eos_figures.plotting import setup_axes, density_panel, label_axes
from eos_figures.stats import hist2d
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts

OUT = REPO + '/Fig_paper'
os.makedirs(OUT, exist_ok=True)
c = Cuts()
cat = load_catalog(REPO + '/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c)

fig, ax = setup_axes(3, figsize=(10, 3))
for axis, mask_name, title in zip(ax, ['acc_al', 'thick_al', 'thin_al'],
                                  ['accreted', r'high-$\alpha$', r'low-$\alpha$']):
    h, xe, ye = hist2d(cat['lz'][m[mask_name]], 1e-5 * cat['energy'][m[mask_name]],
                       c.lzr, c.enr, c.nlz2, c.nen2)
    im = density_panel(axis, h, xe * 1e-3, ye, percentiles=c.perc_elz, vmin=-0.3)
    im.set_rasterized(True)
    axis.axvline(0, color='k', lw=0.8)
    label_axes(axis, r'$L_z\times 10^{-3}$', r'$E\times 10^{-5}$', title)
ax[2].text(-1.4, -0.55, 'Eos', fontsize=9)

for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/obs_energy_pops.{ext}', bbox_inches='tight')
print('wrote', OUT + '/obs_energy_pops.{pdf,png}')
