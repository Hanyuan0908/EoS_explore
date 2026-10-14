"""Diagnostic: TRACE the gas lane instead of drawing a straight line through it.

fig_paper_gas_metallicity drew "the lane" as the straight segment from the disc
centre to the GS/E centroid.  That says where the two objects are, not where the
gas is.

    usage:  diag_lane_ridge.py [SNAP] [all|sf]        (default 72 all)

FIX THE FRAME FIRST.  The paper figure used to read gas_chem_maps_snap72_v3.npz,
built before the azimuth alignment.  In that frame the GS/E sits 8.2 kpc out of
the projection plane (|y|/r = 0.61) and the x-z view foreshortens the satellite
and the lane by 21 per cent.  CONVENTIONS.md lists this failure explicitly.  This
reads the aligned v4 cache.  Nothing below is meaningful in the old frame.

WHAT HAS TO BE SEPARATED.  Two structures run from the disc down towards the
satellite, and they are easy to confuse:

  the lane      narrow, ~2 kpc across, x ~ -9.4 at z = -11 bending to x ~ -8.6 at
                z = -3, and METAL-RICH: its crest is the [Fe/H] maximum at every
                height, -0.37 rising to -0.10 towards the disc.
  the disc-side broader and DENSER -- it carries the higher Sigma above z = -8
  structure    (8.34 against 8.1 in log Sigma at z = -2) -- at x ~ -5.5, and
                ~0.2 dex more metal-poor.  It is not the lane.

Because the second is the brighter, any estimator built on Sigma itself finds it:
a density-weighted centroid converges onto it, and so does a path that maximises
total log Sigma.  What separates them is that the lane is NARROW, so the score is
ridgeness, -d2(log Sigma)/dx2 on a ~1 kpc scale.

Even on ridgeness, the structure with the larger INTEGRATED score over the whole
height range is still the disc-side one -- a freely fitted curve goes there every
time.  What makes the lane the lane is a boundary condition, not a score: it is
the structure that reaches the satellite.  So the curve is ANCHORED at the GS/E
end, at the best-scoring x within W0 of the straight host-to-GS/E join in the
first slice, and only its shape above that is fitted.

THE CENTRELINE IS A CUBIC IN z, not a free path.  An earlier version walked slice
to slice under a penalty on lateral travel (dynamic programming).  That penalty
is the wrong regularisation here: it charges for a steady tilt exactly as it
charges for a wiggle, and the lane IS steadily tilted -- it runs down from the
disc at x = -4.7 to the satellite at x = -9.4.  Stiff enough to suppress the
jump to the other structure, it was also stiff enough to suppress the tilt, so
the line ran vertically and left the lane near the disc, where it was drawn
across gas that is not there.  A low-order polynomial separates the two: smooth
by construction, free to tilt and bend, with the order setting the stiffness.
Degrees 2 and 3 agree to 0.25 kpc; degree 1 cannot follow the bend and is 1.8 kpc
from the metallicity crest against 0.43 for the cubic.

HOW IT IS CHECKED, without circularity.  The curve is fitted on geometry alone --
the narrowness of the density crest -- and never sees the metallicity.  It is
then compared with the [Fe/H] maximum per slice, which the script prints.  The
two agree to a median 0.43 kpc over the whole lane, which is the real evidence
that the traced line is the metal-rich flow and not an artefact of the estimator.
The degree sensitivity is printed beside it.

WHERE IT STARTS AND STOPS.  It starts Z_MARGIN above the top of the inner GS/E
debris contour -- the same 0.5 level the figure draws -- because there is almost
no gas inside that contour and a line across it marks nothing.  It runs to Z_END,
into the host's gas disc, so the lane is drawn joining the two objects it joins.

THE SF MAP IS NOT TRACED FOR THE PAPER FIGURES.  Star-forming gas is clumpy, so
its transverse profile is a chain of clumps rather than a smooth crest.  Both
paper figures draw the path traced on the ALL-gas map: it is the same lane, and
drawing the same line on both is what lets the two figures be compared.  'sf'
here is the diagnostic that checks the lane lies where the SF gas is.

Writes figures/au18_lane_ridge[_sf].png and out/lane_ridge_snap<NN>[_sf].npz.
"""
import os, sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.optimize import minimize
import config_au18 as C

OUT = C.FIG_DIR
SNAP = int(sys.argv[1]) if len(sys.argv) > 1 else 72
GAS = sys.argv[2] if len(sys.argv) > 2 else 'all'
assert GAS in ('all', 'sf')
TAG = '' if GAS == 'all' else '_sf'
Z_END = -0.5                      # the disc plane, where the lane ends
Z_MARGIN = 0.5                    # start this far above the inner GS/E contour
GSE_LEVEL = 0.5                   # the inner debris contour drawn on the figure
X_LO, X_HI = -16., 2.             # x range the path may explore
W0 = 3.0                          # kpc, half-width of the first-slice anchor window
SMOOTH_PIX = 1.6                  # gaussian smoothing of log Sigma, in pixels
RIDGE_PIX = 4.0                   # ~1 kpc: the width of filament the score looks for
DEG = 3                           # polynomial order of the centreline in z
NSTART = (-1., 0., 1., 2., 4.)    # multi-start, so the fit is not seed-dependent

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
    'figure.dpi': 140, 'savefig.dpi': 200,
})

CACHE = (C.OUT_DIR + f'/gas_chem_maps_snap{SNAP}_v4.npz' if GAS == 'all'
         else C.OUT_DIR + f'/gas_sf_maps_snap{SNAP}.npz')
if not os.path.exists(CACHE):
    raise SystemExit(f'missing {CACHE}; run '
                     + ('diag_gas_chemistry_snap72.py' if GAS == 'all'
                        else f'diag_gas_metallicity_sf.py {SNAP}') + ' first')
c = np.load(CACHE)
W, xe, ze = c['W'], c['xe'], c['ze']
XLIM, ZLIM = float(c['XLIM']), float(c['ZLIM'])
NB = len(xe) - 1
SIG = W / ((2 * XLIM / NB) * (2 * ZLIM / NB))
MET = np.where(SIG > 1.5e7, c['mean_XH_Fe'], np.nan) if GAS == 'all' else c['M']
xc, zc = .5 * (xe[:-1] + xe[1:]), .5 * (ze[:-1] + ze[1:])
dx, dz = xc[1] - xc[0], zc[1] - zc[0]
LS = gaussian_filter(np.log10(np.maximum(SIG, 1e3)), SMOOTH_PIX)
RG = np.clip(-gaussian_filter1d(LS, RIDGE_PIX, axis=0, order=2) / dx ** 2, 0, None)

# the top of the inner GS/E contour, reproducing orbit_tools.density_contours
hg, _, hze = np.histogram2d(c['GSE_x'], c['GSE_z'], bins=70,
                            range=[[-XLIM, XLIM], [-ZLIM, ZLIM]])
hg = gaussian_filter(hg, 1.6)
flat = np.sort(hg.ravel())[::-1]
lev = float(flat[np.searchsorted(np.cumsum(flat) / flat.sum(), GSE_LEVEL)])
hzc = .5 * (hze[:-1] + hze[1:])
Z_TOP = float(hzc[(hg >= lev).any(0)].max())
Z_START = Z_TOP + Z_MARGIN
GX, GZ = float(np.median(c['GSE_x'])), float(np.median(c['GSE_z']))
print(f'{GAS} gas, snapshot {SNAP}.  GS/E centroid ({GX:+.2f}, {GZ:+.2f}) kpc')
print(f'inner GS/E contour ({GSE_LEVEL:g}) tops out at z = {Z_TOP:+.1f}; '
      f'the lane starts {Z_MARGIN:g} kpc above it, at z = {Z_START:+.1f}')

kz = np.flatnonzero((zc >= Z_START) & (zc <= Z_END))
zs = zc[kz]
U = (zs - zs[0]) / (zs[-1] - zs[0])      # 0 at the GS/E end, 1 at the disc
X0 = float(xc[np.argmax(np.where(np.abs(xc - GX * zs[0] / GZ) < W0,
                                 RG[:, kz[0]], -np.inf))])
print(f'first slice anchored at x = {X0:+.2f} kpc '
      f'(the straight join would put it at {GX * zs[0] / GZ:+.2f})')


def curve(a):
    """x(z) for coefficients a, with the constant term fixed to the anchor."""
    return X0 + np.polyval(np.append(np.asarray(a)[::-1], 0.), U)


def score(a):
    x = curve(a)
    return -sum(np.interp(x[i], xc, RG[:, k]) for i, k in enumerate(kz))


def fit(deg):
    best = None
    for a1 in NSTART:
        a = np.zeros(deg)
        a[0] = a1
        r = minimize(score, a, method='Nelder-Mead',
                     options=dict(maxiter=6000, xatol=1e-3, fatol=1e-4))
        if best is None or r.fun < best.fun:
            best = r
    return best


best = fit(DEG)
xs = curve(best.x)

# the [Fe/H] crest per slice -- the independent check, never used in the fit
crest = np.full(len(kz), np.nan)
for i, k in enumerate(kz):
    m = (xc > -13) & (xc < -3)
    v = MET[m, k]
    if not np.all(np.isnan(v)):
        crest[i] = xc[m][np.nanargmax(v)]

print('\nsensitivity to the polynomial order, and the check against the [Fe/H] crest:')
for deg in (1, 2, 3):
    b = fit(deg)
    x = curve(b.x)
    o = np.abs(x - crest)
    o = o[np.isfinite(o)]
    print(f'  deg {deg}: ridgeness {-b.fun:6.2f}  ' + '  '.join(
        f'{zs[i]:+5.1f}: {x[i]:+6.2f}' for i in range(0, len(zs), max(len(zs) // 5, 1)))
        + f'   median |offset| {np.median(o):.2f} kpc')
print('  crest  ' + ' ' * 20 + '  '.join(
    f'{zs[i]:+5.1f}: {crest[i]:+6.2f}' for i in range(0, len(zs), max(len(zs) // 5, 1))))

np.savez(C.OUT_DIR + f'/lane_ridge_snap{SNAP}{TAG}.npz', x=xs, z=zs, deg=DEG,
         coef=best.x, x0=X0, ridge_pix=RIDGE_PIX, smooth_pix=SMOOTH_PIX,
         z_top_gse=Z_TOP)
print(f'\ntraced {len(zs)} slices, z = {zs[0]:+.1f} -> {zs[-1]:+.1f} kpc, '
      f'x = {xs[0]:+.2f} -> {xs[-1]:+.2f} kpc')
d = np.abs(xs - GX * zs / GZ)
print(f'offset from the straight join: median {np.median(d):.2f} kpc, max {d.max():.2f}')

fig, AX = plt.subplots(1, 2, figsize=(13.4, 7.0))
for ax, (fld, cm, kw, lc) in zip(AX, [
        (np.where(SIG > 0, SIG, np.nan), 'Greys',
         dict(norm=LogNorm(vmin=3e7, vmax=4e8)), 'k'),
        (MET, 'viridis', dict(vmin=-.7, vmax=0.), 'w')]):
    ax.pcolormesh(xe, ze, fld.T, cmap=cm, rasterized=True, **kw)
    ax.plot([0, GX], [0, GZ], color='#E8112D', ls='--', lw=1.6,
            label='straight host-to-GS/E join')
    ax.plot(curve(fit(1).x), zs, color=lc, lw=.9, ls=':', alpha=.6,
            label='degree 1: cannot follow the bend')
    ax.plot(xs, zs, color='#00E5FF' if lc == 'k' else '#E8112D', lw=2.4,
            label=f'traced lane (degree {DEG})')
    ax.plot(crest, zs, color='#FF6347', lw=1.2, alpha=.8,
            label='[Fe/H] crest (not used in the fit)')
    ax.axhline(Z_TOP, color=lc, lw=.8, ls='-.', alpha=.6,
               label='top of the inner GS/E contour')
    ax.set(aspect='equal', xlim=(-20, 6), ylim=(-18, 4),
           xlabel='$x$ [kpc]', ylabel='$z$ [kpc]')
    ax.legend(loc='lower left', fontsize=10)
fig.suptitle(f'{"all" if GAS == "all" else "star-forming"} gas, snapshot {SNAP}',
             fontsize=14, y=.97)
fig.tight_layout()
f = f'{OUT}/au18_lane_ridge{TAG}.png'
fig.savefig(f, bbox_inches='tight')
print(f'\nsaved {f}')
