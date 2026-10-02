"""Illustration: what "distance along the lane" means in diag_lane_mixing.

The lane sample is a CYLINDER in 3D, laid on the axis that joins the host centre
to the GS/E stellar centroid:

    u        = unit vector from the host centre to the satellite centroid
    t        = (r . u) / |r_GSE|      fractional distance along that axis
    perp     = |r - (r.u) u|          perpendicular distance from the axis
    selected = perp < R_LANE  and  TLO < t < THI

"Distance along the lane" is t x |r_GSE|, in kpc, measured from the host centre.
The seven bins of diag_lane_mixing are t = 0.2 to 0.9 in steps of 0.1, which at
this snapshot puts their centres at 4.9 to 16.7 kpc.

THE FIGURE IS A PROJECTION, THE SELECTION IS NOT.  The cylinder is drawn here as
a rectangle because only one of its two perpendicular directions lies in the
x-z plane; the other points along the line of sight.  A cell can therefore sit
inside the drawn rectangle and still be outside the cylinder, by being displaced
in y.  The grey background is the all-gas surface density, which IS projected.

Why a cylinder on that axis rather than something fitted to the gas: the axis is
defined by two positions that are measured independently of the gas (the halo
centre and the clean-debris centroid), so the coordinate cannot be accused of
having been drawn around the structure it then reports.

Reads the cached maps; no snapshot load.  Writes figures/au18_lane_geometry.png.
"""
import os, sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
import config_au18 as C
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import orbit_tools as OT
import auriga_public as ap          # noqa: F401  (config import parity)

SNAP = 72
R_LANE, TLO, THI, NT = 3.5, 0.20, 0.90, 7
# the 3D centroid used by diag_lane_mixing, after the azimuth rotation
AX = np.array([-13.42, -14.42])     # (x, z); the y component is ~0 by construction
OUT = C.FIG_DIR
CACHE = C.OUT_DIR + f'/gas_chem_maps_snap{SNAP}_v3.npz'

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 13.5, 'axes.labelsize': 15,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 150, 'savefig.dpi': 200,
})

c = np.load(CACHE)
W, xe, ze = c['W'], c['xe'], c['ze']
AREA = float((xe[1] - xe[0]) * (ze[1] - ze[0]))
SIG = np.where(W > 0, W / AREA, np.nan)
XLIM, ZLIM = float(abs(xe[-1])), float(abs(ze[-1]))

Lax = float(np.linalg.norm(AX))
u = AX / Lax
pvec = np.array([-u[1], u[0]])                 # in-plane perpendicular
print(f'snapshot {SNAP}: satellite centroid ({AX[0]:+.2f}, {AX[1]:+.2f}) kpc, '
      f'|r| = {Lax:.2f} kpc')
edges = np.linspace(TLO, THI, NT + 1)
print(f'bin centres along the lane [kpc]: '
      + ', '.join(f'{.5 * (edges[i] + edges[i+1]) * Lax:.1f}' for i in range(NT)))

GREY = ListedColormap(plt.get_cmap('Greys')(np.linspace(0., .62, 256)))
fig, ax = plt.subplots(figsize=(8.6, 7.4))
ax.pcolormesh(xe, ze, SIG.T, cmap=GREY, norm=LogNorm(vmin=1e7, vmax=1e9),
              rasterized=True, zorder=0)
OT.density_contours(ax, c['GSE_x'], c['GSE_z'], [[-XLIM, XLIM], [-ZLIM, ZLIM]],
                    '#8E24AA', levels=(0.9, 0.5), bins=70, smooth=1.6, lw=2.0)

cL = '#E8112D'
# the axis, dashed outside the selected range and solid inside it
ax.plot([0, AX[0]], [0, AX[1]], color='k', ls=':', lw=1.4, zorder=3)
A, B = TLO * AX, THI * AX
ax.plot([A[0], B[0]], [A[1], B[1]], color=cL, lw=2.2, zorder=4)

# the cylinder wall, as it projects
for sgn in (+1, -1):
    p0, p1 = A + sgn * R_LANE * pvec, B + sgn * R_LANE * pvec
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=cL, lw=2.0, zorder=4)
for t in (TLO, THI):
    q = t * AX
    ax.plot([(q + R_LANE * pvec)[0], (q - R_LANE * pvec)[0]],
            [(q + R_LANE * pvec)[1], (q - R_LANE * pvec)[1]],
            color=cL, lw=2.0, zorder=4)

# the seven bin walls, and a label on each bin centre
for t in edges[1:-1]:
    q = t * AX
    ax.plot([(q + R_LANE * pvec)[0], (q - R_LANE * pvec)[0]],
            [(q + R_LANE * pvec)[1], (q - R_LANE * pvec)[1]],
            color=cL, lw=1.0, ls='--', alpha=.8, zorder=4)
# keep the labels upright: the axis points down-left, so its raw angle would set
# the text upside down
ANG = np.degrees(np.arctan2(u[1], u[0]))
if ANG > 90 or ANG < -90:
    ANG += 180
for i in range(NT):
    tc = .5 * (edges[i] + edges[i + 1])
    q = tc * AX + (R_LANE + 1.5) * pvec
    ax.text(q[0], q[1], f'{tc * Lax:.0f}', color=cL, fontsize=11.5, ha='center',
            va='center', zorder=6, rotation=ANG,
            bbox=dict(fc='white', ec='none', alpha=.8, pad=1.2))

# the two ends, measured independently of the gas
ax.plot(0, 0, '*', color='k', ms=19, mec='w', mew=1.1, zorder=7)
ax.plot(AX[0], AX[1], '*', color='#8E24AA', ms=19, mec='w', mew=1.1, zorder=7)
bb = dict(fc='white', ec='none', alpha=.85, pad=2.2)
ax.text(1.6, 1.6, 'host centre\n($t = 0$)', fontsize=12, va='bottom', bbox=bb, zorder=7)
ax.text(AX[0] - 1.2, AX[1] - 1.6, 'GS/E centroid\n($t = 1$)', fontsize=12,
        ha='right', va='top', color='#8E24AA', bbox=bb, zorder=7)

# the perpendicular radius, drawn where the cylinder is widest on the page
qm = .55 * AX
ax.annotate('', xy=tuple(qm + R_LANE * pvec), xytext=tuple(qm),
            arrowprops=dict(arrowstyle='<->', color=cL, lw=1.6), zorder=6)
ax.text(*(qm + .5 * R_LANE * pvec + np.array([-1.0, 0.9])),
        f'$R_{{\\rm lane}}$ = {R_LANE} kpc', color=cL, fontsize=12, ha='right',
        bbox=bb, zorder=7)
ax.annotate(f'$t$ = {TLO}', xy=tuple(A - (R_LANE + .6) * pvec), fontsize=11.5,
            color=cL, ha='center', va='center', bbox=bb, zorder=7)
ax.annotate(f'$t$ = {THI}', xy=tuple(B - (R_LANE + .6) * pvec), fontsize=11.5,
            color=cL, ha='center', va='center', bbox=bb, zorder=7)
ax.text(.975, .025,
        'labels: distance along the lane [kpc]\n'
        'selection is a 3D cylinder; this is its projection',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=11, bbox=bb,
        zorder=7)
ax.set(aspect='equal', xlim=(-XLIM, XLIM), ylim=(-ZLIM, ZLIM),
       xlabel='$x$ [kpc]', ylabel='$z$ [kpc]')
fig.tight_layout(pad=.4)
f = f'{OUT}/au18_lane_geometry.png'
fig.savefig(f, bbox_inches='tight')
print(f'\nsaved {f}')
