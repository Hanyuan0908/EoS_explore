"""Diagnostic, NOT a paper figure: au18_gas_metallicity at an arbitrary snapshot.

Exactly the two panels of fig_paper_gas_metallicity -- gas surface density, and
mass-weighted mean gas [Fe/H] in the same pixels -- but for any snapshot, so the
state at the pericentre can be compared with the snapshots leading up to it.

    usage:  diag_gas_metallicity_snap.py [SNAP]        (default 72)

Three things are read from the cache rather than hard-coded, because they move
between snapshots and the paper version pins them to snapshot 72:

  * the frame (XLIM, ZLIM), which diag_gas_chemistry_snap72.py sizes to contain
    the GS/E -- at snapshot 73 the satellite sits at r = 34 kpc and a fixed frame
    crops it;
  * the GS/E centroid, which sets where the lane is drawn;
  * the pixel area behind the surface density, which depends on the frame.

The frames are NOT the same size between snapshots and cannot be: the satellite
is 90 kpc from the host at snapshot 70 and 20 kpc at 72, so a frame that holds it
at 72 loses it entirely at 70.  Each panel states its own frame, and the pixel
area behind Sigma_gas is taken from the stored pixel edges, so the surface
densities are comparable even though the fields of view are not.

The colour limits VMIN, VMAX, SIG_LO, SIG_HI and the blanking threshold SMIN are
deliberately kept FIXED across snapshots.  Rescaling each panel to its own data would make the
sequence unreadable as a sequence: the question these figures answer is whether
the satellite gas changes, and that needs one scale.

Run diag_gas_chemistry_snap72.py <SNAP> first to build the cache.
Writes figures/au18_gas_metallicity_snap<NN>.png -- figures/, not Fig_paper/.
"""
import os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import config_au18 as C
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import orbit_tools as OT

OUT = C.FIG_DIR
os.makedirs(OUT, exist_ok=True)
SNAP = int(sys.argv[1]) if len(sys.argv) > 1 else 72
VMIN, VMAX = -0.7, 0.0                      # fixed across snapshots, on purpose
SIG_LO, SIG_HI = 1e7, 1e9                   # so is the surface-density scale
SMIN = 1.5e7
NB = 240
GSE_AB = (8., 5.)
# v4 adds XLIM/ZLIM as stored keys; v3 does not, and snapshot 72 only has v3.
# Prefer v4 and fall back, deriving the frame from the pixel edges either way.
CACHE = next((f for f in (C.OUT_DIR + f'/gas_chem_maps_snap{SNAP}_v4.npz',
                          C.OUT_DIR + f'/gas_chem_maps_snap{SNAP}_v3.npz')
              if os.path.exists(f)), None)
if CACHE is None:
    raise SystemExit(f'no cache for snapshot {SNAP}; run '
                     f'"diag_gas_chemistry_snap72.py {SNAP}" first to build it')
st = np.load(C.OUT_DIR + '/snapshot_times.npz')
TSNAP = float(st['t_snap'][np.flatnonzero(st['snaps'] == SNAP)[0]])

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
XLIM, ZLIM = float(abs(xe[-1])), float(abs(ze[-1]))
GSE_CEN = (float(np.median(c['GSE_x'])), float(np.median(c['GSE_z'])))
AREA = float((xe[1] - xe[0]) * (ze[1] - ze[0]))   # from the edges, not from NB
print(f'snapshot {SNAP}, t = {TSNAP:.3f} Gyr; frame {XLIM:.0f} x {ZLIM:.0f} kpc; '
      f'GS/E centroid ({GSE_CEN[0]:+.1f}, {GSE_CEN[1]:+.1f}) kpc, '
      f'r = {np.hypot(*GSE_CEN):.1f}')
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
                   norm=LogNorm(vmin=SIG_LO, vmax=SIG_HI), rasterized=True)
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
ax.text(.5, .965, f'snapshot {SNAP},  $t$ = {TSNAP:.2f} Gyr', transform=ax.transAxes,
        va='top', ha='center', fontsize=13,
        bbox=dict(fc='white', ec='none', alpha=.8, pad=2))

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
f = f'{OUT}/au18_gas_metallicity_snap{SNAP}.png'
fig.savefig(f, bbox_inches='tight')
print(f'\nsaved {f}')
