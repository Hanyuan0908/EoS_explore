"""Publication figure (observational): three panels assembled verbatim from the
existing reference figures, with one addition (a colourbar on the left panel).

  (a) "Magnesium"     -- the middle panel of figures_repro/01_fig1_energy_mg_al.png
  (b) "Aluminium"     -- the right  panel of figures_repro/01_fig1_energy_mg_al.png
  (c) "Mean [Al/Fe]"  -- the middle panel of figures_repro/01_fig4_alfe_3pops.png

Each panel is reproduced exactly (same reference helpers, ranges, lines, labels,
titles as `eos_figures.figures.plot_energy_mg_al` / `plot_alfe_3pops`).  The ONLY
change is a colourbar added to panel (a), inset in its lower-left empty space.
Panel (c) keeps its original inset [Al/Fe] colourbar; panel (b) is unchanged.

Data: data_repro/our_apogee_dr17_lite_ann.fits.gz (in-repo, portable).
Writes Fig_paper/obs_mg_al_meanal.pdf and .png.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

REPO = '/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore'
sys.path.insert(0, REPO + '/eos-figures')
# importing eos_figures.plotting applies the reference repo's own rcParams (serif,
# size 10, ticks-in) -- do NOT override, so the panels match the originals exactly.
from eos_figures.plotting import density_panel, value_panel, label_axes
from eos_figures.stats import hist2d, stat2d
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts

OUT = REPO + '/Fig_paper'
os.makedirs(OUT, exist_ok=True)
c = Cuts()
cat = load_catalog(REPO + '/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c)

fig, ax = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)

# ===== (a) Magnesium  (plot_energy_mg_al, ax[1]) =====
h, xe, ye = hist2d(cat['fe_h'][m['base']], cat['mg_fe'][m['base']], c.fehr, c.mgfer, c.nfeh, c.nmg)
im0 = density_panel(ax[0], h, xe, ye, percentiles=(2, 98))
im0.set_rasterized(True)
xx = np.array(c.fehr)
ax[0].plot(xx, c.slope_acc * xx + c.inter_acc, 'w-', lw=1.1)
ax[0].plot(xx, c.slope_acc * xx + c.inter_acc, 'k--', lw=0.8)
ax[0].plot(xx, c.slope_acc2 * xx + c.inter_acc2, 'w-', lw=1.1)
ax[0].plot(xx, c.slope_acc2 * xx + c.inter_acc2, 'k:', lw=1.0)
ax[0].text(-1.8, -0.02, 'accreted', fontsize=8)
ax[0].text(-0.8, 0.32, r'high-$\alpha$', color='w', fontsize=8)
ax[0].text(-0.35, 0.06, r'low-$\alpha$', color='w', fontsize=8, rotation=-30)
label_axes(ax[0], '[Fe/H]', '[Mg/Fe]', 'Magnesium')
# --- THE ONLY CHANGE: colourbar in the lower-left empty space of panel (a) ---
cax0 = ax[0].inset_axes([0.08, 0.13, 0.38, 0.035])
cb0 = fig.colorbar(im0, cax=cax0, orientation='horizontal')
cb0.set_label(r'$\log_{10} N$', fontsize=8)
cb0.ax.xaxis.set_label_position('top')
cb0.ax.xaxis.set_ticks_position('bottom')
cb0.ax.tick_params(labelsize=7, length=2)

# ===== (b) Aluminium  (plot_energy_mg_al, ax[2]) =====
h, xe, ye = hist2d(cat['fe_h'][m['base']], cat['al_fe'][m['base']], c.fehr, c.alfer, c.nfeh, c.nal)
im1 = density_panel(ax[1], h, xe, ye, percentiles=c.perc1)
im1.set_rasterized(True)
ax[1].axhline(c.alfe_cut, color='k', ls='--', lw=0.8)
label_axes(ax[1], '[Fe/H]', '[Al/Fe]', 'Aluminium')

# ===== (c) Mean [Al/Fe]  (plot_alfe_3pops, ax[1]) =====
h0, xe, ye = hist2d(cat['fe_h'][m['base_en']], cat['mg_fe'][m['base_en']], c.fehr2, c.mgfer2, c.nfeh2, c.nmg2)
mean_al, xe, ye = stat2d(cat['fe_h'][m['base_en']], cat['mg_fe'][m['base_en']], cat['al_fe'][m['base_en']],
                         c.fehr2, c.mgfer2, c.nfeh2, c.nmg2, statistic='mean')
im_mid = value_panel(ax[2], mean_al, xe, ye, -0.2, 0.27, mask=h0 <= 1, cmap='RdYlBu_r')
im_mid.set_rasterized(True)
cax = ax[2].inset_axes([0.12, 0.10, 0.56, 0.035])
cb = fig.colorbar(im_mid, cax=cax, orientation='horizontal')
cb.set_label('[Al/Fe]', fontsize=8)
cb.ax.xaxis.set_label_position('top')
cb.ax.xaxis.set_ticks_position('bottom')
cb.ax.tick_params(labelsize=7, length=2)
xx2 = np.array(c.fehr2)
ax[2].plot(xx2, c.slope_acc * xx2 + c.inter_acc, 'k--', lw=0.8)
ax[2].plot(xx2, c.slope_acc2 * xx2 + c.inter_acc2, 'k:', lw=1.0)
ax[2].set_xlim(c.fehr2); ax[2].set_ylim(c.mgfer2)
label_axes(ax[2], '[Fe/H]', '[Mg/Fe]', 'Mean [Al/Fe]')

for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/obs_mg_al_meanal.{ext}', bbox_inches='tight')
print('wrote', OUT + '/obs_mg_al_meanal.{pdf,png}')
