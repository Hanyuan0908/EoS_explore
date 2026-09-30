"""Shared disc-frame helper for the Au18 edge-on figures.

The disc axis is well defined -- aligning z to it puts the net angular momentum
within a few degrees of z at every snapshot -- but the AZIMUTH around that axis
is not.  It comes from the in-plane principal axes of a nearly axisymmetric disc,
which are close to degenerate, so it points somewhere different at every
snapshot.  For an edge-on figure that matters: whatever lies along the projection
axis y is foreshortened, and at snapshot 73 the GS/E sits at |y|/r = 0.59, so a
plain x-z view puts a satellite at r = 34 kpc only 28 kpc from the centre and
shortens the gas lane with it.

`align_azimuth` removes that freedom by rotating about the disc axis until a
chosen point -- here the GS/E centroid -- lies in the x-z plane.  The projection
then contains the satellite and the lane at their true lengths, and different
snapshots become comparable to each other.
"""
import numpy as np


def align_azimuth(target_xz, x_sign=-1.):
    """Rotation about z putting `target_xz` (a 3-vector) in the x-z plane.

    Returns a 3x3 matrix.  x_sign chooses which side of the panel the target
    lands on: -1 puts it at negative x, matching the earlier snapshot-72 figures.
    """
    ang = np.arctan2(target_xz[1], target_xz[0])
    if x_sign < 0:
        ang -= np.pi
    c, s = np.cos(-ang), np.sin(-ang)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def gse_centroid(pos, ids, gse_ids):
    """Median position of the clean GS/E debris, in the frame `pos` is given in."""
    o = np.argsort(ids); ss = ids[o]
    p = np.searchsorted(ss, gse_ids)
    ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == gse_ids)
    G = pos[o[p[ok]]]
    return np.median(G, axis=0), G, o, ss
