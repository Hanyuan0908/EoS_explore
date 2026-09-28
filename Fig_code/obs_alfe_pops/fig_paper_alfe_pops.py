"""Publication figure (observational): [Al/Fe]-[Fe/H] for the three populations,
reproduced exactly from eos_figures.figures.plot_alfe_pops
(== figures_repro/01_fig2_alfe_pops.png).

Top row: column-normalised [Al/Fe]-[Fe/H] density for accreted / high-a / low-a,
with the accreted diagonal and the in-situ Al cut. Bottom row: the same planes
coloured by median V_tan (colourbar on the high-a panel); the GS/E, Aurora,
Splash+high-a disk, Eos and low-a disk features are labelled.

Data: data_repro/our_apogee_dr17_lite_ann.fits.gz (in-repo, portable).
Writes Fig_paper/obs_alfe_pops.pdf and .png.
"""
import os
import sys
import numpy as np

REPO = '/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore'
sys.path.insert(0, REPO + '/eos-figures')
from eos_figures.plotting import setup_axes, density_panel, value_panel, label_axes
from eos_figures.stats import hist2d, stat2d
from eos_figures.figures import _idl_low_density_mask
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts

OUT = REPO + '/Fig_paper'
os.makedirs(OUT, exist_ok=True)
c = Cuts()
cat = load_catalog(REPO + '/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c)

fig, ax = setup_axes(3, nrows=3, figsize=(10, 9))
specs = [('acc', 'accreted', c.perc),
         ('thick', r'high-$\alpha$', c.perc2),
         ('thin', r'low-$\alpha$', c.perc2)]
hist_cache, mask_cache = {}, {}
# top row: column-normalised density
for i, (mask_name, title, perc) in enumerate(specs):
    h, xe, ye = hist2d(cat['fe_h'][m[mask_name]], cat['al_fe'][m[mask_name]],
                       c.fehr, c.alfer, c.nfeh, c.nal2, normalize='x')
    hist_cache[mask_name] = (h, xe, ye)
    mask_cache[mask_name] = _idl_low_density_mask(h, perc, c.white_lim)
    im = density_panel(ax[i], h, xe, ye, percentiles=perc); im.set_rasterized(True)
    ax[i].plot(c.fehr, np.array(c.fehr) * c.kalfe + c.offalfe, 'k--', lw=0.8)
    ax[i].axhline(c.alfe_cut, color='k', ls='--', lw=0.8)
    ax[i].set_xlim(c.fehr); ax[i].set_ylim(c.alfer)
    label_axes(ax[i], '[Fe/H]', '[Al/Fe]', title)
# bottom row: coloured by median V_tan
for i, (mask_name, title, _) in enumerate(specs, start=3):
    h, xe, ye = hist_cache[mask_name]
    vmask = (m[mask_name] & np.isfinite(cat['galvt'])
             & (cat['galvt'] >= c.vtanr[0]) & (cat['galvt'] <= c.vtanr[1]))
    med, _, _ = stat2d(cat['fe_h'][vmask], cat['al_fe'][vmask], cat['galvt'][vmask],
                       c.fehr, c.alfer, c.nfeh, c.nal2)
    h_med, _, _ = hist2d(cat['fe_h'][vmask], cat['al_fe'][vmask], c.fehr, c.alfer, c.nfeh, c.nal2)
    med = np.nan_to_num(med, nan=0.0); med[h_med <= 2] = 0.0
    im = value_panel(ax[i], med, xe, ye, *c.mm_vtan, mask=mask_cache[mask_name], cmap='RdYlBu_r')
    im.set_rasterized(True)
    if i == 4:
        im4 = im
    ax[i].plot(c.fehr, np.array(c.fehr) * c.kalfe + c.offalfe, 'k--', lw=0.8)
    ax[i].axhline(c.alfe_cut, color='k', ls='--', lw=0.8)
    ax[i].set_xlim(c.fehr); ax[i].set_ylim(c.alfer)
    label_axes(ax[i], '[Fe/H]', '[Al/Fe]')          # no titles on the second row
ax[3].text(-1.5, 0.01, 'GS/E', fontsize=9)
ax[4].text(-1.75, 0.0, 'Aurora', fontsize=9, rotation=60)
ax[4].text(-1.25, 0.44, r'Splash+high-$\alpha$ disk', fontsize=8)
ax[5].text(-1.3, -0.1, 'Eos', fontsize=9, rotation=60)
ax[5].text(-0.75, 0.3, r'low-$\alpha$ disk', fontsize=8)

# V_phi colourbar inset in the empty bottom of the high-alpha V_phi panel (no external
# colourbar, so it does not open a gap between the rows)
cax4 = ax[4].inset_axes([0.28, 0.11, 0.58, 0.045])
cb4 = fig.colorbar(im4, cax=cax4, orientation='horizontal')
cb4.set_label(r'$V_\phi$ [km/s]', fontsize=9)
cb4.ax.xaxis.set_label_position('top')
cb4.ax.xaxis.set_ticks_position('bottom')
cb4.ax.tick_params(labelsize=8, length=2)

# third row: E-Lz of the three populations (as in obs_energy_pops). On the accreted
# panel the accreted stars are shown BEFORE the kinematic cut (chemistry only) with a
# dark-grey mask over |Lz|>500 kpc km/s -- the region the accreted selection removes.
lz = np.asarray(cat['lz'], float); energy = np.asarray(cat['energy'], float)
_base = np.asarray(m['base'], bool)
_feh = np.asarray(cat['fe_h'], float); _mg = np.asarray(cat['mg_fe'], float)
acc_chem = _base & (_mg < c.slope_acc * _feh + c.inter_acc) & (_mg < c.slope_acc2 * _feh + c.inter_acc2)
elz = [(acc_chem, True), (np.asarray(m['thick'], bool), False), (np.asarray(m['thin'], bool), False)]
for i, (mk, shade) in enumerate(elz, start=6):
    h, xe3, ye3 = hist2d(lz[mk], 1e-5 * energy[mk], c.lzr, c.enr, c.nlz2, c.nen2)
    im = density_panel(ax[i], h, xe3 * 1e-3, ye3, percentiles=c.perc_elz, vmin=-0.3)
    im.set_rasterized(True)
    ax[i].axvline(0, color='k', lw=0.8)
    if shade:
        ax[i].axvspan(0.5, c.lzr[1] * 1e-3, color='0.35', alpha=0.55, zorder=4)
        ax[i].axvspan(c.lzr[0] * 1e-3, -0.5, color='0.35', alpha=0.55, zorder=4)
    label_axes(ax[i], r'$L_z\times 10^{-3}$', r'$E\times 10^{-5}$')

for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/obs_alfe_pops.{ext}', bbox_inches='tight')
print('wrote', OUT + '/obs_alfe_pops.{pdf,png}')
