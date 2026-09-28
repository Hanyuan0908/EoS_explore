"""Non-publish: [Mg/Fe] vs stellar age for the LAMOST Xiang subgiant sample,
TWO panels -- Milky Way (in-situ) and GS/E (accreted).

Each panel: column-normalised (per age) log density in the age-[Mg/Fe] plane
(Greys), with the 5th and 95th percentile tracks of [Mg/Fe] at each age (both red)
-- the obs_nfe_pops house style, here with age on the x-axis.

Populations via eos_figures.make_masks on the LAMOST cache (same cuts as APOGEE):
  MW in-situ = thin_al | thick_al  (NOT split into high/low-alpha)
  GS/E       = acc_al  (chem-clean accreted: [Fe/H]<-0.5, [Al/Fe]<-0.12)
Ages: Xiang MSTO, quality cut sigma_age/age<0.3 and age<14 Gyr. DD-Payne [Mg/Fe].
NB: GS/E subgiant ages are unreliable (metal-poor, small sample) -- trust only the
old-age end; the young percentile tracks there are age-error/contamination.

Data: data_repro/our_lamost_subgiant_ddpayne.fits.gz. Output figures_repro/01_lamost_mgfe_age.png.
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
    'font.size': 15, 'axes.labelsize': 19, 'axes.titlesize': 18,
    'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 13,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})

FIG = Path(REPO + '/figures_repro')
c = Cuts()
cat = load_catalog(REPO + '/data_repro/our_lamost_subgiant_ddpayne.fits.gz')
m = make_masks(cat, c)
age = np.asarray(cat['age'], float); aerr = np.asarray(cat['age_model_error'], float)
mg = np.asarray(cat['mg_fe'], float)
rel_ok = np.isfinite(age) & np.isfinite(aerr) & (age > 0) & (age < 14) & (aerr / age < 0.3) & np.isfinite(mg)

mw = (np.asarray(m['thin_al'], bool) | np.asarray(m['thick_al'], bool)) & rel_ok   # MW in-situ (combined)
gse = np.asarray(m['acc_al'], bool) & rel_ok                                       # GS/E accreted (chem-clean)
RED = '#E8112D'
AGER = (0, 14); MGR = (-0.1, 0.45); NX, NY = 28, 28

specs = [('Milky Way (in-situ)', mw, 0.5, 30), ('GS/E (accreted)', gse, 1.5, 15)]


def pctl_tracks(sel, dage, minn):
    edges = np.arange(AGER[0], AGER[1] + 1e-9, dage); cen = 0.5 * (edges[:-1] + edges[1:])
    p5 = np.full(len(cen), np.nan); p95 = np.full(len(cen), np.nan)
    for i in range(len(cen)):
        yy = mg[sel & (age >= edges[i]) & (age < edges[i + 1])]
        if yy.size >= minn:
            p5[i], p95[i] = np.percentile(yy, [5, 95])
    return cen, p5, p95


# raw log-count density, per-panel stretch (MW and GS/E differ ~300x in N, so a
# shared/column-normalised scale would either wash out GS/E or amplify its noise).
fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.4), sharex=True, sharey=True, constrained_layout=True)

for j, (title, sel, dage, minn) in enumerate(specs):
    h, xe, ye = hist2d(age[sel], mg[sel], AGER, MGR, NX, NY)      # raw counts
    im = log_image(h); fin = im[np.isfinite(im)]
    vmin, vmax = finite_percentile(fin, c.perc2)
    imsh = ax[j].imshow(im.T, origin='lower', extent=[xe[0], xe[-1], ye[0], ye[-1]],
                        aspect='auto', interpolation='nearest', cmap='Greys', vmin=vmin, vmax=vmax)
    imsh.set_rasterized(True)
    cen, p5, p95 = pctl_tracks(sel, dage, minn)
    ax[j].plot(cen, p95, color=RED, lw=2.4, zorder=6)
    ax[j].plot(cen, p5, color=RED, lw=2.4, zorder=6)
    ax[j].set_title(f'{title}   ($N={int(sel.sum())}$)')
    ax[j].set_xlabel('age [Gyr]  (Xiang MSTO)')
    ax[j].set_xlim(*AGER); ax[j].set_ylim(*MGR)
    cb = fig.colorbar(imsh, ax=ax[j], location='right', pad=0.02, aspect=26)
    cb.set_label(r'$\log_{10} N$'); cb.ax.tick_params(length=3)
ax[0].set_ylabel('[Mg/Fe]  (DD-Payne)')

fig.savefig(FIG / '01_lamost_mgfe_age.png', bbox_inches='tight')
print('wrote', FIG / '01_lamost_mgfe_age.png')
print(f'N: MW in-situ={int(mw.sum())}, GS/E={int(gse.sum())}')
