"""Publication figure: the gas bridge at the GS/E pericentre, and its metallicity.

Two panels, edge-on in the disc frame at snapshot 72 (t = 4.99 Gyr):

  (a)  gas surface density -- the lane joining the host disc to the satellite
  (b)  mass-weighted mean gas [Fe/H] in the same pixels

The point: the lane the halo-born stars form in is chemically the host's gas on a
metallicity gradient, not a distinct satellite reservoir.  [Fe/H] falls smoothly
along it with no discontinuity where a second reservoir would begin.  The
companion result is that every [X/Fe] is flat to within 0.05 dex between disc,
lane and GS/E (diag_gas_chemistry_snap72.py).

NO REGION LABELS ON THE MAP.  An earlier version annotated a median [Fe/H] for
the disc, lane and GS/E, which was misleading: those were unweighted medians over
all pixels in each mask, and the disc mask (|z| < 2, |x| < 8 kpc) holds far more
faint metal-poor pixels than bright core ones, so it read -0.29 while the visible
yellow core is ~0.0.  The numbers were right for their masks and wrong for what
the eye compares them against.  Quote region values from
diag_gas_chemistry_snap72.py, which states its weighting, not off this map.

Panel (b) blanks pixels below SMIN in gas surface density (31.8 per cent of pixels
shown, holding 76.8 per cent of the gas).

viridis, not a diverging map: [Fe/H] has no meaningful midpoint and a red-blue
ramp invites the eye to read one.

Reads the moments cached by diag_gas_chemistry_snap72.py; run that first.
Writes Fig_paper/au18_gas_metallicity.pdf and .png.
"""
import os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import config_au18 as C
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import orbit_tools as OT

OUT = '/data/hz420-2/EoS_explore/Fig_paper'
os.makedirs(OUT, exist_ok=True)
SNAP = 72
VMIN, VMAX = -0.7, 0.0
SMIN = 1.5e7
XLIM, ZLIM, NB = 30., 24., 240
GSE_CEN, GSE_AB = (-11., -14.), (8., 5.)
CACHE = C.OUT_DIR + f'/gas_chem_maps_snap{SNAP}_v3.npz'
if not os.path.exists(CACHE):
    raise SystemExit(f'missing {CACHE}; run diag_gas_chemistry_snap{SNAP}.py first')

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 13.5, 'axes.labelsize': 15,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12.5,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})

c = np.load(CACHE)
W, xe, ze = c['W'], c['xe'], c['ze']
AREA = (2 * XLIM / NB) * (2 * ZLIM / NB)
SIG = W / AREA
M = c['mean_XH_Fe'].copy()
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
print(f'[Fe/H]: disc {dm:+.2f}, lane {lm:+.2f}, GS/E {gg:+.2f}')
print(f'Sigma > {SMIN:.1e}: {100 * (~faint).mean():.1f}% of pixels, '
      f'{100 * W[~faint].sum() / W.sum():.1f}% of the gas')

# Explicit axes rectangles.  With aspect='equal' matplotlib shrinks each axes to
# fit its allotted box and centres it, so subplots(hspace=0) still leaves a gap;
# placing the panels by hand is the only way to make them touch exactly.
FW = 7.8
AXL, AXW = .115, .715                       # left edge and width, figure fractions
axw_in = FW * AXW
axh_in = axw_in * (2 * ZLIM) / (2 * XLIM)   # equal aspect
FH = 2 * axh_in + 1.02                      # + margins
fig = plt.figure(figsize=(FW, FH))
b0, h = .055 * (9.8 / FH), axh_in / FH
axes = [fig.add_axes([AXL, b0 + h, AXW, h]), fig.add_axes([AXL, b0, AXW, h])]
GAP = .012
cax_a = fig.add_axes([AXL + AXW + .018, b0 + h + GAP, .026, h - GAP])
cax_b = fig.add_axes([AXL + AXW + .018, b0, .026, h - GAP])
RNG = [[-XLIM, XLIM], [-ZLIM, ZLIM]]

# --- (a) gas surface density -------------------------------------------------
ax = axes[0]
S = np.where(SIG > 0, SIG, np.nan)
im = ax.pcolormesh(xe, ze, S.T, cmap='Greys',
                   norm=LogNorm(vmin=np.nanpercentile(S, 45),
                                vmax=np.nanpercentile(S, 99.9)), rasterized=True)
cb = fig.colorbar(im, cax=cax_a)
cb.set_label(r'$\Sigma_{\rm gas}$ [M$_\odot$ kpc$^{-2}$]')
# mark where panel (b) stops showing pixels
OT.density_contours(ax, c['GSE_x'], c['GSE_z'], RNG, '#8E24AA',
                    levels=(0.9, 0.5), bins=70, smooth=1.6, lw=2.2)
ax.plot([], [], color='#8E24AA', lw=2.2, label='GS/E stellar debris')
ax.plot([0, GSE_CEN[0]], [0, GSE_CEN[1]], color='k', ls='--', lw=1.8, alpha=.85,
        label='the lane')
ax.legend(loc='lower right', handlelength=1.5, borderpad=.35)
ax.text(.03, .965, '(a)', transform=ax.transAxes, va='top', fontsize=16,
        fontweight='bold')

# --- (b) gas metallicity -----------------------------------------------------
ax = axes[1]
im = ax.pcolormesh(xe, ze, M.T, cmap='viridis', vmin=VMIN, vmax=VMAX, rasterized=True)
cb = fig.colorbar(im, cax=cax_b)
cb.set_label('[Fe/H] of the gas')
OT.density_contours(ax, c['GSE_x'], c['GSE_z'], RNG, 'w',
                    levels=(0.9, 0.5), bins=70, smooth=1.6, lw=2.2)
ax.plot([0, GSE_CEN[0]], [0, GSE_CEN[1]], color='w', ls='--', lw=1.8, alpha=.9)
ax.text(.03, .965, '(b)', transform=ax.transAxes, va='top', fontsize=16,
        fontweight='bold')

for a in axes:
    a.set(aspect='equal', xlim=(-XLIM, XLIM), ylim=(-ZLIM, ZLIM), ylabel='$z$ [kpc]')
axes[0].tick_params(labelbottom=False)
axes[1].set_xlabel('$x$ [kpc]')
# Drop the tick that sits on the shared edge, where panel (b)'s top tick already
# is.  set_yticks can disturb the view limits, so restore them afterwards.
keep = [t for t in axes[0].get_yticks() if -ZLIM + 1 < t < ZLIM - 1]
axes[0].set_yticks(keep)
axes[0].set_ylim(-ZLIM, ZLIM)
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/au18_gas_metallicity.{ext}', bbox_inches='tight')
print(f'\nsaved {OUT}/au18_gas_metallicity.pdf and .png')
