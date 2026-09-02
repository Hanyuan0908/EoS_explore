"""EXPERIMENT: the gas [Fe/H] map alone, at the GS/E pericentre.

The [Fe/H] panel of diag_gas_chemistry_snap72.py on its own and at full size, on
a fixed colour range of -0.8 to -0.1 rather than one derived from the disc-to-GS/E
transition.  A fixed range is the honest choice once the scale is being read as a
number rather than as a relative gradient, and -0.8 to -0.1 brackets the three
reference values: disc -0.30, lane -0.48, GS/E -0.69.

Mass-weighted mean per pixel, from the moments cached by
diag_gas_chemistry_snap72.py in out/gas_chem_maps_snap<NN>_v3.npz; run that first
if the cache is missing.  Violet contours enclose 50 and 90 per cent of the clean
GS/E stellar debris, the dashed line is the lane corridor.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import config_au18 as C
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import orbit_tools as OT

SNAP = int(sys.argv[1]) if len(sys.argv) > 1 else 72
VMIN, VMAX = -0.8, -0.1
# Blank pixels below a gas surface density, to drop the diffuse speckle without
# hollowing out the structure.  At 1.5e7 Msun/kpc^2 about a third of pixels
# survive and hold ~80 per cent of the gas in the frame; 3e7 was cleaner but cut
# into the fainter envelope around the lane.  Override on the command line.
SMIN = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5e7
NB, MMIN = 240, 5e5
GSE_AB = (8., 5.)
CACHE = C.OUT_DIR + f'/gas_chem_maps_snap{SNAP}_v4.npz'
if not os.path.exists(CACHE):
    raise SystemExit(f'missing {CACHE}; run diag_gas_chemistry_snap{SNAP}.py first')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 13, 'axes.labelsize': 14, 'xtick.labelsize': 12,
    'ytick.labelsize': 12, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'figure.dpi': 150, 'savefig.dpi': 200,
})

c = np.load(CACHE)
W, xe, ze = c['W'], c['xe'], c['ze']
XLIM, ZLIM = float(c['XLIM']), float(c['ZLIM'])   # frame is sized to the GS/E
GSE_CEN = (float(np.median(c['GSE_x'])), float(np.median(c['GSE_z'])))
print(f'GS/E centroid at snapshot {SNAP}: ({GSE_CEN[0]:.1f}, {GSE_CEN[1]:.1f}) kpc')
M = c['mean_XH_Fe'].copy()
AREA = (2 * XLIM / NB) * (2 * ZLIM / NB)
SIG = c['W'] / AREA
faint = SIG < SMIN
M[faint] = np.nan
xc, zc = .5 * (xe[:-1] + xe[1:]), .5 * (ze[:-1] + ze[1:])
X, Z = np.meshgrid(xc, zc, indexing='ij')
ok = np.isfinite(M)
DISC = (np.abs(Z) < 2) & (np.abs(X) < 8)
GSEM = ((X - GSE_CEN[0]) / GSE_AB[0]) ** 2 + ((Z - GSE_CEN[1]) / GSE_AB[1]) ** 2 < 1
t = np.clip((X * GSE_CEN[0] + Z * GSE_CEN[1]) / (GSE_CEN[0] ** 2 + GSE_CEN[1] ** 2), 0, 1)
LANE = (np.hypot(X - t * GSE_CEN[0], Z - t * GSE_CEN[1]) < 3.5) & (t > .25) & (t < .85)
md = lambda m: float(np.nanmedian(M[m & ok]))
dm, lm, gg = md(DISC), md(LANE), md(GSEM)
print(f'[Fe/H] medians:  disc {dm:+.2f}   lane {lm:+.2f}   GS/E {gg:+.2f}')
print(f'Sigma_gas > {SMIN:.1e} Msun/kpc^2: {100 * (~faint).mean():.1f}% of pixels shown, '
      f'holding {100 * c["W"][~faint].sum() / c["W"].sum():.1f}% of the gas')
print(f'colour range fixed at {VMIN} to {VMAX}; '
      f'{100 * np.nanmean(M[ok] < VMIN):.1f}% of shown pixels below, '
      f'{100 * np.nanmean(M[ok] > VMAX):.1f}% above')

fig, ax = plt.subplots(figsize=(10.2, 8.0))
# viridis, not a diverging map: [Fe/H] has no meaningful midpoint, and a
# red-blue ramp invites the eye to read one where none exists.
im = ax.pcolormesh(xe, ze, M.T, cmap='viridis', vmin=VMIN, vmax=VMAX, rasterized=True)
cb = fig.colorbar(im, ax=ax, pad=.015, extend='both')
cb.set_label('[Fe/H] of the gas (mass-weighted mean)')
# White over viridis: the old violet contour would vanish into its dark end.
OT.density_contours(ax, c['GSE_x'], c['GSE_z'], [[-XLIM, XLIM], [-ZLIM, ZLIM]],
                    'w', levels=(0.9, 0.5), bins=70, smooth=1.6, lw=2.2)
ax.plot([0, GSE_CEN[0]], [0, GSE_CEN[1]], color='w', ls='--', lw=1.8, alpha=.9)
for (px, pz), lab, val in [((2.5, 1.5), 'disc', dm),
                           ((-6.5, -7.5), 'lane', lm),
                           ((-11., -16.5), 'GS/E', gg)]:
    ax.annotate(f'{lab}: {val:+.2f}', (px, pz), fontsize=12.5, ha='center',
                bbox=dict(fc='white', ec='none', alpha=.85, pad=2.2))
ax.set(aspect='equal', xlim=(-XLIM, XLIM), ylim=(-ZLIM, ZLIM),
       xlabel='$x$ [kpc]', ylabel='$z$ [kpc]')
_st = np.load(C.OUT_DIR + '/snapshot_times.npz')
TSNAP = float(_st['t_snap'][list(_st['snaps']).index(SNAP)])
ax.set_title(f'Au18 snapshot {SNAP} ($t$ = {TSNAP:.2f} Gyr): gas metallicity at the GS/E '
             f'pericentre\n' + r'cells with $\Sigma_{\rm gas} < $'
             + f'{SMIN:.1e}' + r' M$_\odot$ kpc$^{-2}$ not shown', fontsize=13.5)
fig.tight_layout()
out = C.FIG_DIR + f'/diag_gas_metallicity_snap{SNAP}.png'
fig.savefig(out)
print('saved', out)
