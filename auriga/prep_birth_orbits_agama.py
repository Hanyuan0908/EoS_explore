"""Circularity of every in-situ Au18 star from the AGAMA CylSpline potentials.

For each snapshot the nearest-in-time potential (prep_potentials_agama.py) supplies
E -> R_circ(E) -> L_circ(E), and eps = L_z / L_circ(E).  Unlike the two earlier
estimators this is normalised by the potential, not by the star distribution, so
eps is bounded by ~1 and means the same thing at t = 2 Gyr and t = 13.8 Gyr.

EVERY snapshot is measured in its OWN frame: the potential is built with its
symmetry axis along that snapshot's disc axis, and L_z is taken about that same
axis.  This is the only self-consistent choice.  L_z about any other axis is not
conserved in an axisymmetric potential and L_circ(E) is only defined about the
symmetry axis, so mixing a contemporary potential with the z=0 axis -- which an
earlier version of this script did -- produces a number that is not a circularity
at all.

The consequence is worth stating, because the disc axis swings ~94 degrees between
t = 5 Gyr and z = 0.  A star born on a disc orbit in the OLD disc that keeps its
angular momentum while the gas disc reforms in a new plane genuinely ends up on an
inclined orbit relative to the disc that exists today: that is real misalignment,
not a bookkeeping artefact.  Equally, a star born misaligned with the old stellar
disc but aligned with the plane the disc is about to adopt is genuinely born on a
disc orbit of the FUTURE disc.  Both are physical, and both are only visible if
each epoch is measured against its own disc.

Note the residual ambiguity: the disc axis here is the angular momentum of the
STARS within 10 kpc, i.e. the old stellar disc.  During the merger the gas may
already be settling into a different plane, so the "contemporary disc" is not
uniquely defined for ~1 Gyr around coalescence.  axis_gap records how far the
assigned potential's axis is from the star snapshot's own.
"""
import gc, os
import numpy as np
import agama
import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod

agama.setUnits(mass=1, length=1, velocity=1)
PDIR = C.OUT_DIR + '/potentials'
idx = np.load(PDIR + '/index.npz')
POT_SNAP, POT_T, POT_AX = idx['snaps'], idx['t'], idx['axis']

st = np.load(C.OUT_DIR + '/snapshot_times.npz')
SNAPS, T_SNAP = st['snaps'], st['t_snap']
_cache = {}


def get_pot(sn_t):
    """The potential whose snapshot is closest in time to this one."""
    k = int(POT_SNAP[np.argmin(np.abs(POT_T - sn_t))])
    if k not in _cache:
        _cache.clear()                                  # one at a time; they are big
        _cache[k] = agama.Potential(f'{PDIR}/pot_{k:03d}.ini')
    return k, _cache[k]


def rotation_to(axis):
    z = axis / np.linalg.norm(axis)
    tmp = np.array([1., 0., 0.])
    if abs(np.dot(tmp, z)) > .9: tmp = np.array([0., 1., 0.])
    x = np.cross(tmp, z); x /= np.linalg.norm(x)
    return np.vstack([x, np.cross(z, x), z])


def load_stars(sn):
    s = snap_mod.load_snapshot(sn, 4, snappath=C.SIM_DIR,
        loadlist=['ParticleIDs', 'Coordinates', 'Velocities', 'Masses',
                  'GFM_StellarFormationTime'])
    real = s.data['GFM_StellarFormationTime'] > 0
    sf = sub_mod.subfind(sn, directory=C.SIM_DIR, loadlist=['GroupFirstSub', 'SubhaloPos'])
    cen = sf.data['SubhaloPos'][int(sf.data['GroupFirstSub'][0])]
    x = (s.data['Coordinates'][real] - cen) * 1000.
    v = s.data['Velocities'][real]; m = s.data['Masses'][real]
    ids = s.data['ParticleIDs'][real]
    r = np.sqrt((x * x).sum(1)); inn = r < 10.
    v = v - np.average(v[inn], axis=0, weights=m[inn])
    J = (m[inn, None] * np.cross(x[inn], v[inn])).sum(0)
    del s; gc.collect()
    return ids, x, v, J / np.linalg.norm(J)


def circularity(x, v, axis, pot):
    """eps = Lz/Lcirc(E) about `axis`, plus the pieces, in the potential's units."""
    R = rotation_to(axis)
    xr, vr = x @ R.T, v @ R.T
    Rc_ = np.hypot(xr[:, 0], xr[:, 1])
    Lz = xr[:, 0] * vr[:, 1] - xr[:, 1] * vr[:, 0]
    phi = pot.potential(xr)
    E = .5 * (vr * vr).sum(1) + phi
    ok = np.isfinite(E) & (E < 0)
    rc = np.full(len(E), np.nan); lc = np.full(len(E), np.nan)
    if ok.any():
        rci = pot.Rcirc(E=E[ok])
        good = np.isfinite(rci) & (rci > 0)
        ii = np.flatnonzero(ok)[good]; rci = rci[good]
        fr = pot.force(np.column_stack([rci, np.zeros_like(rci), np.zeros_like(rci)]))[:, 0]
        vc2 = -rci * fr
        rc[ii] = rci
        lc[ii] = np.where(vc2 > 0, rci * np.sqrt(np.abs(vc2)), np.nan)
    return Lz / lc, E, Lz, lc, Rc_, xr[:, 2]


cat = np.load(C.OUT_DIR + '/z0_insitu_catalog.npz')
ids_all, tform_all = cat['ids'], cat['tform']
keep = tform_all >= T_SNAP[0]
ids_k, tform_k = ids_all[keep], tform_all[keep]
assigned = np.searchsorted(T_SNAP, tform_k)
print(f'{len(ids_k):,} in-situ stars measurable near birth', flush=True)

KEYS = ('eps_birth', 'E_birth', 'Lz_birth', 'Lcirc_birth', 'R_birth', 'z_birth')
out = {k: np.full(len(ids_k), np.nan, np.float32) for k in KEYS}
pot_used = np.full(len(ids_k), -1, np.int16)
axis_gap = np.full(len(SNAPS), np.nan)
axis_now = np.full((len(SNAPS), 3), np.nan)

for k, sn in enumerate(SNAPS):
    sel = np.flatnonzero(assigned == k)
    if not len(sel): continue
    ids, x, v, ax_now = load_stars(int(sn))
    pk, pot = get_pot(float(T_SNAP[k]))
    axis_gap[k] = np.degrees(np.arccos(np.clip(
        ax_now @ POT_AX[np.argmax(POT_SNAP == pk)], -1, 1)))
    axis_now[k] = ax_now
    en, E, Lz, lc, Rcyl, zcyl = circularity(x, v, ax_now, pot)
    o = np.argsort(ids); ss = ids[o]
    want = ids_k[sel]
    p = np.searchsorted(ss, want)
    ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == want)
    ix = o[p[ok]]; dst = sel[ok]
    for key, arr in [('eps_birth', en), ('E_birth', E), ('Lz_birth', Lz),
                     ('Lcirc_birth', lc), ('R_birth', Rcyl), ('z_birth', zcyl)]:
        out[key][dst] = arr[ix]
    pot_used[dst] = pk
    print(f'snap {sn:3d} t={T_SNAP[k]:5.2f}  pot {pk:3d}  axis gap {axis_gap[k]:5.1f} deg  '
          f'born {len(sel):6,}  matched {ok.sum():6,}', flush=True)
    del ids, x, v, en, E, Lz, lc; gc.collect()

# z = 0, same machinery, every star.
ids, x, v, ax0 = load_stars(127)
pk, pot = get_pot(float(T_SNAP[-1]))
e0 = circularity(x, v, ax0, pot)
o = np.argsort(ids); ss = ids[o]
p = np.searchsorted(ss, ids_k)
ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == ids_k)
ix = o[p[ok]]
z0 = {k: np.full(len(ids_k), np.nan, np.float32)
      for k in ('eps_z0', 'E_z0', 'Lz_z0', 'Lcirc_z0', 'R_z0')}
for key, arr in [('eps_z0', e0[0]), ('E_z0', e0[1]), ('Lz_z0', e0[2]),
                 ('Lcirc_z0', e0[3]), ('R_z0', e0[4])]:
    z0[key][ok] = arr[ix]
print(f'z=0 pass: matched {ok.sum():,}/{len(ids_k):,}', flush=True)

np.savez(C.OUT_DIR + '/birth_orbits_agama.npz', ids=ids_k, tform=tform_k,
         pot_used=pot_used, snaps=SNAPS, t_snap=T_SNAP, axis_gap=axis_gap,
         axis_now=axis_now, axis_z0=ax0, **out, **z0)
n = np.isfinite(out['eps_birth'])
print(f'\nmeasured {n.sum():,}/{len(ids_k):,}; '
      f'eps>1: {100 * (out["eps_birth"][n] > 1).mean():.2f}%  '
      f'max {np.nanmax(out["eps_birth"]):.2f}')
print('saved', C.OUT_DIR + '/birth_orbits_agama.npz')
