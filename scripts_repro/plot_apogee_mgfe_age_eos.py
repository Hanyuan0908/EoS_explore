"""[Mg/Fe] vs stellar age (AstroNN) -- MW in-situ and GS/E density panels
(companion to plot_apogee_mgfe_age.py) with the Eos stars overplotted.

Eos (canonical cut: halo & low-a wedge, -0.9<[Fe/H]<-0.2, [Al/Fe]>-0.12) is in-situ,
so it is overplotted on the MW panel, split into its two branches:
  metal-poor (alpha-rich, upper)  -> magenta
  metal-rich (alpha-poor, lower)  -> cyan
Age quality: sigma_age/age<0.3. Output figures_repro/01_apogee_mgfe_age_eos.png.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

REPO = '/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore'
sys.path.insert(0, REPO + '/eos-figures')
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
from eos_figures.stats import hist2d, log_image, finite_percentile

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 15, 'axes.labelsize': 19, 'axes.titlesize': 18, 'legend.fontsize': 12,
    'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})
FIG = Path(REPO + '/figures_repro')
c = Cuts()
cat = load_catalog(REPO + '/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c)
age = np.asarray(cat['age'], float); aerr = np.asarray(cat['age_model_error'], float)
mg = np.asarray(cat['mg_fe'], float); feh = np.asarray(cat['fe_h'], float); al = np.asarray(cat['al_fe'], float)
lz = np.asarray(cat['lz'], float); rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float)
ecc = (rap - rperi) / (rap + rperi)
rel_ok = np.isfinite(age) & np.isfinite(aerr) & (age > 0) & (age < 14) & (aerr / age < 0.3) & np.isfinite(mg)

mw = (np.asarray(m['thin_al'], bool) | np.asarray(m['thick_al'], bool)) & rel_ok
gse = np.asarray(m['acc_al'], bool) & rel_ok
RED = '#E8112D'; CMP, CMR = 'magenta', 'cyan'
AGER = (0, 14); MGR = (-0.1, 0.45); NX, NY = 28, 28

# canonical Eos + Davies branch split
halo = np.asarray(m['base'], bool) & ((ecc > 0.7) | (lz < 0))
eos = halo & (feh > -0.9) & (feh < -0.2) & (mg > c.slope_acc * feh + c.inter_acc) \
    & (mg < c.slope_acc2 * feh + c.inter_acc2) & (al > c.alfe_cut) & rel_ok
eos_mp = eos & (mg > 0.317 * feh + 0.353)      # alpha-rich / upper = metal-poor
eos_mr = eos & (mg <= 0.317 * feh + 0.353)     # alpha-poor / lower = metal-rich

specs = [('Milky Way (in-situ)', mw, 0.5, 30), ('GS/E (accreted)', gse, 1.5, 15)]


def pctl_tracks(sel, dage, minn):
    edges = np.arange(AGER[0], AGER[1] + 1e-9, dage); cen = 0.5 * (edges[:-1] + edges[1:])
    p5 = np.full(len(cen), np.nan); p95 = np.full(len(cen), np.nan)
    for i in range(len(cen)):
        yy = mg[sel & (age >= edges[i]) & (age < edges[i + 1])]
        if yy.size >= minn:
            p5[i], p95[i] = np.percentile(yy, [5, 95])
    return cen, p5, p95


fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.4), sharex=True, sharey=True, constrained_layout=True)
for j, (title, sel, dage, minn) in enumerate(specs):
    h, xe, ye = hist2d(age[sel], mg[sel], AGER, MGR, NX, NY)
    im = log_image(h); vmin, vmax = finite_percentile(im[np.isfinite(im)], c.perc2)
    imsh = ax[j].imshow(im.T, origin='lower', extent=[xe[0], xe[-1], ye[0], ye[-1]],
                        aspect='auto', interpolation='nearest', cmap='Greys', vmin=vmin, vmax=vmax)
    imsh.set_rasterized(True)
    cen, p5, p95 = pctl_tracks(sel, dage, minn)
    ax[j].plot(cen, p95, color=RED, lw=2.4, zorder=6)
    ax[j].plot(cen, p5, color=RED, lw=2.4, zorder=6)
    ax[j].set_title(f'{title}   ($N={int(sel.sum())}$)')
    ax[j].set_xlabel('age [Gyr]  (AstroNN)')
    ax[j].set_xlim(*AGER); ax[j].set_ylim(*MGR)
    cb = fig.colorbar(imsh, ax=ax[j], location='right', pad=0.02, aspect=26)
    cb.set_label(r'$\log_{10} N$'); cb.ax.tick_params(length=3)
ax[0].set_ylabel('[Mg/Fe]  (APOGEE)')

# overplot Eos (in-situ) on the MW panel, split by branch
ax[0].scatter(age[eos_mp], mg[eos_mp], s=22, c=CMP, edgecolors='k', linewidths=0.3, zorder=8,
              label=fr'Eos metal-poor ($n={int(eos_mp.sum())}$, $\tilde\tau={np.median(age[eos_mp]):.1f}$)')
ax[0].scatter(age[eos_mr], mg[eos_mr], s=22, c=CMR, edgecolors='k', linewidths=0.3, zorder=8,
              label=fr'Eos metal-rich ($n={int(eos_mr.sum())}$, $\tilde\tau={np.median(age[eos_mr]):.1f}$)')
ax[0].legend(loc='lower right', framealpha=0.85)

fig.savefig(FIG / '01_apogee_mgfe_age_eos.png', bbox_inches='tight')
print('wrote', FIG / '01_apogee_mgfe_age_eos.png')
print(f'Eos metal-poor n={int(eos_mp.sum())} med_age={np.median(age[eos_mp]):.1f}; '
      f'metal-rich n={int(eos_mr.sum())} med_age={np.median(age[eos_mr]):.1f}')
