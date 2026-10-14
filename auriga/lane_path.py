"""The traced centreline of the gas lane, and the mask that goes with it.

The lane was drawn and measured as the straight segment from the disc centre to
the GS/E centroid.  That is a statement about where the two objects are, not
about where the gas is.  In the azimuth-aligned frame the lane runs almost
vertically, from x = -9.4 kpc at z = -10.7 to x = -8.6 at z = -0.7, while the
straight join sweeps from x = -10.0 to -0.7 across the same heights: the two are
3.4 kpc apart on average and 8.0 kpc apart at the top.

This module loads the ridge traced by diag_lane_ridge.py and builds the mask from
it, so the line drawn on a figure and the numbers quoted for "the lane" are the
same object.  Both consumers -- fig_paper_gas_metallicity.py and
diag_gas_chemistry_snap72.py -- go through here for exactly that reason.

The mask keeps pixels within RADIUS of the path.  NO ARC-LENGTH TRIM.  The
straight-line version kept only t = 0.25-0.85 because its endpoints were the disc
centre and the GS/E centroid, so both ends lay inside an object rather than in the
lane.  The traced path already begins just above the inner GS/E debris contour and
ends where the filament loses its crest against the disc, so trimming it again
would only throw away lane.

RADIUS is left at the 3.5 kpc the straight-line mask used, so the lane statistics
stay comparable with the ones already quoted, and it turns out to be about right:
the lane's transverse FWHM is 6.1 kpc in all gas and 5.4 kpc in star-forming gas,
i.e. a half-width near 3 kpc.  The median still moves with it -- [Fe/H] = -0.38,
-0.43, -0.44, -0.45 at radius 1.5, 2.0, 2.5, 3.5 kpc for all gas, and -0.31,
-0.36, -0.40, -0.42 for star-forming gas -- so quote the radius with the number.
"""
import numpy as np

RADIUS = 3.5          # kpc from the centreline, as the straight-line mask used
FLO, FHI = 0., 1.     # the whole path; see below


def load(out_dir, snap=72):
    """(x, z) of the traced lane centreline, ordered GS/E -> disc."""
    import os
    f = f'{out_dir}/lane_ridge_snap{snap}.npz'
    if not os.path.exists(f):
        raise SystemExit(f'missing {f}; run diag_lane_ridge.py first')
    d = np.load(f)
    return d['x'], d['z']


def arclength(px, pz):
    s = np.concatenate([[0.], np.cumsum(np.hypot(np.diff(px), np.diff(pz)))])
    return s / s[-1]


def mask(X, Z, px, pz, radius=RADIUS, flo=FLO, fhi=FHI):
    """Pixels within `radius` of the path, over the trimmed part of its length.

    Distance is to the polyline's VERTICES rather than to its segments.  The
    trace is sampled every 0.2 kpc, so the two differ by at most 0.1 kpc -- far
    below the 3.5 kpc radius -- and the vertex form also gives the arc-length
    position for free.
    """
    f = arclength(px, pz)
    d = np.hypot(X[..., None] - px, Z[..., None] - pz)
    k = np.argmin(d, axis=-1)
    return (np.take_along_axis(d, k[..., None], -1)[..., 0] < radius) & \
           (f[k] > flo) & (f[k] < fhi)
