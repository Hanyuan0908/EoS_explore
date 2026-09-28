"""Non-publish: reproduction of Ciuca et al. (2024) Fig. 3 -- the age-metallicity
plane coloured by [Mg/Fe], in four Galactocentric-radius bins, using BINGO ages.

Four panels (4<R<6, 6<R<8, 8<R<10, 10<R<12 kpc): [Fe/H] vs age tau, scatter
coloured by [Mg/Fe] (jet, -0.1..0.5), with black density contours and vertical
dashed lines at tau=10 and 12 Gyr (the GS/E window). x-axis reversed (old left).

Reproduces Ciuca+24 as closely as possible: uses the RAW BINGO age
(age = 10**pred_logAge), which extends to ~18 Gyr -- NOT the lowess-corrected age
(which caps at ~10.5 Gyr). ASPCAP [Fe/H] (FE_H_1), [Mg/Fe] (MG_FE_1), galr all from
the BINGO catalogue itself. Quality cuts follow Ciuca+24 Sec. 2: SNR>100,
4000<Teff<5500, 1<logg<3.5, sigma(log tau)<=0.2, [Fe/H]>-1, and drop age<8 Gyr &
[Mg/Fe]>0.2 (merged binaries). Reproduces their sample count (68,430 in 4<R<12 vs 68,360).

Data (Mac-only): APOGEE/APOGEE_DR17_bingoages.fits. Output figures_repro/01_ciuca_amr_bingo.png.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
from pathlib import Path
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from astropy.table import Table
warnings.filterwarnings('ignore')

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 14, 'axes.labelsize': 17, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})

BINGO = '/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE/APOGEE_DR17_bingoages.fits'
FIG = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/figures_repro')

t = Table.read(BINGO)
age = np.asarray(t['age'], float)                       # RAW BINGO age = 10**pred_logAge (to ~18 Gyr)
feh = np.asarray(t['FE_H_1'], float)
mg = np.asarray(t['MG_FE_1'], float)
galr = np.asarray(t['galr'], float)
# Ciuca+24 quality cuts (their Sec. 2): SNR>100, 4000<Teff<5500, 1<logg<3.5,
# BINGO age precision sigma(log tau)<=0.2, [Fe/H]>-1, and a merged-binary filter
# (drop age<8 Gyr with [Mg/Fe]>0.2). The catalogue is already RGB/RC pre-selected.
snr = np.asarray(t['SNR'], float)
teff = np.asarray(t['TEFF_1'], float)
logg = np.asarray(t['LOGG_1'], float)
slogage = np.asarray(t['pred_logAge_std'], float)       # BINGO age uncertainty in log10
ok = (np.asarray(t['MG_FE_FLAG']) == 0) & (np.asarray(t['FE_H_FLAG']) == 0) \
     & np.isfinite(age) & np.isfinite(feh) & np.isfinite(mg) & np.isfinite(galr) \
     & (snr > 100) & (teff > 4000) & (teff < 5500) & (logg > 1) & (logg < 3.5) \
     & (slogage <= 0.2) & (feh > -1) \
     & ~((age < 8) & (mg > 0.2))                        # merged-binary filter

AGER = (0, 18); FEHR = (-1.15, 0.75)
VMIN, VMAX = -0.1, 0.5
RBINS = [(4, 6), (6, 8), (8, 10), (10, 12)]


def contours(ax, x, y):
    H, xe, ye = np.histogram2d(x, y, bins=[45, 40], range=[AGER, FEHR])
    Hs = gaussian_filter(H, 1.0)
    if Hs.max() <= 0:
        return
    # levels enclosing ~90% and ~50% of the mass
    flat = np.sort(Hs.ravel())[::-1]; csum = np.cumsum(flat) / flat.sum()
    lv = [flat[np.searchsorted(csum, f)] for f in (0.90, 0.50)]
    xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
    ax.contour(xc, yc, Hs.T, levels=sorted(set(lv)), colors='k', linewidths=0.8, zorder=4)


fig, axes = plt.subplots(2, 2, figsize=(13.6, 7.8), sharex=True, sharey=True, constrained_layout=True)
sc = None
for ax, (r0, r1) in zip(axes.ravel(), RBINS):
    s = ok & (galr >= r0) & (galr < r1)
    order = np.argsort(mg[s])                           # plot high-[Mg/Fe] on top
    xs, ys, cs = age[s][order], feh[s][order], mg[s][order]
    sc = ax.scatter(xs, ys, c=cs, s=2.0, cmap='jet', vmin=VMIN, vmax=VMAX,
                    edgecolors='none', alpha=0.75, rasterized=True, zorder=2)
    contours(ax, age[s], feh[s])
    for xv in (10, 12):
        ax.axvline(xv, color='0.35', ls='--', lw=1.0, zorder=3)
    ax.text(0.97, 0.06, f'${r0} < R < {r1}$ kpc', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=15,
            bbox=dict(fc='white', ec='none', alpha=0.8, pad=2))
    ax.set_xlim(AGER[1], AGER[0]); ax.set_ylim(*FEHR)    # reversed x (old on left)

# feature labels in the first panel, following Ciuca+24
axes[0, 0].text(11.0, 0.42, 'GGS', fontsize=13, color='k')
axes[0, 0].text(16.2, -0.32, 'Babi', fontsize=13, color='k')
axes[0, 0].text(12.3, -0.62, 'Dip', fontsize=13, color='k')

for ax in axes[1, :]:
    ax.set_xlabel(r'$\tau$ [Gyr]  (BINGO)')
for ax in axes[:, 0]:
    ax.set_ylabel('[Fe/H] [dex]')

cb = fig.colorbar(sc, ax=axes, location='right', pad=0.02, aspect=30, extend='both')
cb.set_label('[Mg/Fe] [dex]')

fig.savefig(FIG / '01_ciuca_amr_bingo.png', bbox_inches='tight')
print('wrote', FIG / '01_ciuca_amr_bingo.png')
for r0, r1 in RBINS:
    print(f'  {r0}<R<{r1}: N={int((ok & (galr>=r0) & (galr<r1)).sum())}')
