"""Publication figure: the gas bridge at the GS/E pericentre, STAR-FORMING gas.

The companion to fig_paper_gas_metallicity, panel for panel, with one change:
every cell here is flagged star-forming.  Two panels, edge-on in the disc frame
at snapshot 72 (t = 4.99 Gyr):

  (a)  star-forming gas surface density
  (b)  its mass-weighted mean [Fe/H] in the same pixels

WHY BOTH FIGURES EXIST.  The all-gas map is dominated by material that will never
make a star: 60 per cent of the gas within 50 kpc of the host is not star-forming,
and it is the diffuse, metal-poor component, so the all-gas metallicity map
averages the lane together with a reservoir irrelevant to the burst.  This one
shows only the gas eligible to form stars -- the gas whose composition the
halo-born population will actually inherit.  The argument the pair makes is that
the lane is the host's gas on a metallicity gradient either way: restricting to
star-forming cells does not reveal a second, distinct reservoir.

STAR-FORMING = Auriga's own StarFormationRate > 0 flag, the definition
prep_gas_disc_au18.py uses.  Grand et al. (2017) quote the threshold as
n_H > 0.13 cm^-3; the minimum density among flagged cells measured in this run is
0.107 cm^-3, consistent once the fixed X_H = 0.76 behind that estimate is allowed
for.  It is NOT a temperature cut, deliberately: Auriga puts star-forming gas on
the Springel & Hernquist effective equation of state, so its InternalEnergy is a
pressure floor and not a temperature.  See diag_gas_metallicity_sf.py.

NO SURFACE-DENSITY BLANKING, unlike the all-gas figure.  That blanks faint pixels
because its frame is mostly diffuse material; star-forming gas is dense by
construction, so the problem does not arise.  A pixel's [Fe/H] is drawn once it
holds MMIN = 1e5 Msun; 20.7 per cent of pixels qualify and they hold 99.2 per
cent of the star-forming gas.

THE LANE IS THE ONE TRACED ON THE ALL-GAS MAP, deliberately.  Star-forming gas is
clumpy, so its transverse profile is a chain of clumps rather than a smooth crest
and a trace run on it wanders between them.  Drawing the same line on both
figures is also what lets them be compared: it is one lane seen two ways, not two
measurements of two things.

Frame, grid and GS/E debris are identical to the all-gas figure (same azimuth
alignment, same snapshot), so the two can be laid side by side.

Reads the maps cached by diag_gas_metallicity_sf.py and the lane centreline from
diag_lane_ridge.py; run both first.
Writes Fig_paper/au18_gas_metallicity_sf.pdf and .png.
"""
import os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import config_au18 as C
import lane_path as LP
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import orbit_tools as OT

OUT = '/data/hz420-2/EoS_explore/Fig_paper'
os.makedirs(OUT, exist_ok=True)
SNAP = 72
VMIN, VMAX = -0.7, 0.0
SIG_LO, SIG_HI = 1e6, 1e9
GSE_AB = (8., 5.)
# THE GS/E CONTOUR COLOUR, used in BOTH panels of BOTH gas figures.
#
# The house GS/E colour is violet #8E24AA, and it fails on a viridis [Fe/H] map:
# viridis's low end IS dark violet, so the contour vanishes wherever it crosses
# metal-poor gas -- which is most of the satellite, exactly where it matters.
# What was tried, on both panels at once:
#   white        invisible in the SF figure, whose panel (b) is 79 per cent blank
#   grey / black read on white and on yellow, lost on the dark-violet end, and
#                on panel (a) they compete with the Greys data itself
#   red          the most legible, but CONVENTIONS.md forbids a warm red or
#                orange for the GS/E (it clashes with born-hot tomato) and
#                #E8112D is already the kinematic-cut-line colour
#   cyan / green sit inside viridis's own teal-to-green range
# #D500F9 is the house violet raised in lightness and saturation: the same hue,
# so nothing about the convention's meaning changes, but now well clear of
# viridis's dark end and of the greyscale in panel (a).
CGSE = '#D500F9'
CACHE = C.OUT_DIR + f'/gas_sf_maps_snap{SNAP}.npz'
if not os.path.exists(CACHE):
    raise SystemExit(f'missing {CACHE}; run diag_gas_metallicity_sf.py {SNAP} first')

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
W, M, xe, ze = c['W'], c['M'], c['xe'], c['ze']
XLIM, ZLIM = float(c['XLIM']), float(c['ZLIM'])
NB = len(xe) - 1
SIG = W / ((2 * XLIM / NB) * (2 * ZLIM / NB))
SIG = np.where(SIG > 0, SIG, np.nan)
GSE_CEN = (float(np.median(c['GSE_x'])), float(np.median(c['GSE_z'])))

xc, zc = .5 * (xe[:-1] + xe[1:]), .5 * (ze[:-1] + ze[1:])
X, Z = np.meshgrid(xc, zc, indexing='ij')
LX, LZ = LP.load(C.OUT_DIR, SNAP)
ok = np.isfinite(M)
DISC = (np.abs(Z) < 2) & (np.abs(X) < 8)
GSEM = ((X - GSE_CEN[0]) / GSE_AB[0]) ** 2 + ((Z - GSE_CEN[1]) / GSE_AB[1]) ** 2 < 1
LANE = LP.mask(X, Z, LX, LZ)
md = lambda m: float(np.nanmedian(M[m & ok]))
print(f'star-forming gas, snapshot {SNAP}')
print(f'  [Fe/H]: disc {md(DISC):+.2f}, lane {md(LANE):+.2f}, GS/E {md(GSEM):+.2f}')
print(f'  pixels with a drawn [Fe/H]: {100 * ok.mean():.1f} %')

# Explicit axes rectangles, as in the all-gas figure: with aspect='equal'
# matplotlib shrinks each axes inside its gridspec slot and centres it, so
# hspace=0 still leaves a gap.
FW = 7.8
AXL, AXW = .115, .715
axw_in = FW * AXW
axh_in = axw_in * ZLIM / XLIM
FH = 2 * axh_in + 1.02
fig = plt.figure(figsize=(FW, FH))
b0, h = .055 * (9.8 / FH), axh_in / FH
axes = [fig.add_axes([AXL, b0 + h, AXW, h]), fig.add_axes([AXL, b0, AXW, h])]
GAP = .012
cax_a = fig.add_axes([AXL + AXW + .018, b0 + h + GAP, .026, h - GAP])
cax_b = fig.add_axes([AXL + AXW + .018, b0, .026, h - GAP])
RNG = [[-XLIM, XLIM], [-ZLIM, ZLIM]]

# --- (a) star-forming gas surface density ------------------------------------
ax = axes[0]
im = ax.pcolormesh(xe, ze, SIG.T, cmap='Greys',
                   norm=LogNorm(vmin=SIG_LO, vmax=SIG_HI), rasterized=True)
cb = fig.colorbar(im, cax=cax_a)
cb.set_label(r'$\Sigma_{\rm gas,\,SF}$ [M$_\odot$ kpc$^{-2}$]')
OT.density_contours(ax, c['GSE_x'], c['GSE_z'], RNG, CGSE,
                    levels=(0.9, 0.5), bins=70, smooth=1.6, lw=2.2)
ax.plot([], [], color=CGSE, lw=2.2, label='GS/E stellar debris')
ax.plot(LX, LZ, color='k', ls='--', lw=1.8, alpha=.85, label='the lane')
ax.legend(loc='lower right', handlelength=1.5, borderpad=.35)
ax.text(.03, .965, '(a)', transform=ax.transAxes, va='top', fontsize=16,
        fontweight='bold')

# --- (b) star-forming gas metallicity ----------------------------------------
ax = axes[1]
im = ax.pcolormesh(xe, ze, M.T, cmap='viridis', vmin=VMIN, vmax=VMAX,
                   rasterized=True)
cb = fig.colorbar(im, cax=cax_b)
cb.set_label('[Fe/H] of the star-forming gas')
# NOT white, unlike the all-gas figure.  Panel (b) here is mostly blank: only
# 20.7 per cent of pixels hold enough star-forming gas to be drawn, so white
# markings disappear into the background rather than standing out against it.
# The GS/E contour uses the same colour in both panels (CGSE, set above), and
# the lane keeps the black of panel (a), so the two panels now read the same.
OT.density_contours(ax, c['GSE_x'], c['GSE_z'], RNG, CGSE,
                    levels=(0.9, 0.5), bins=70, smooth=1.6, lw=2.2)
ax.plot(LX, LZ, color='k', ls='--', lw=1.8, alpha=.9)
ax.text(.03, .965, '(b)', transform=ax.transAxes, va='top', fontsize=16,
        fontweight='bold')

for a in axes:
    a.set(aspect='equal', xlim=(-XLIM, XLIM), ylim=(-ZLIM, ZLIM), ylabel='$z$ [kpc]')
axes[0].tick_params(labelbottom=False)
axes[1].set_xlabel('$x$ [kpc]')
keep = [t for t in axes[0].get_yticks() if -ZLIM + 1 < t < ZLIM - 1]
axes[0].set_yticks(keep)
axes[0].set_ylim(-ZLIM, ZLIM)
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/au18_gas_metallicity_sf.{ext}', bbox_inches='tight')
print(f'\nsaved {OUT}/au18_gas_metallicity_sf.pdf and .png')
