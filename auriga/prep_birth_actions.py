"""Birth actions (J_r, J_z, J_phi) and circularity for every in-situ Au18 star.

Uses the potentials from prep_potentials_ref.py, which unlike the earlier set
support AGAMA's ActionFinder.  For each star, measured in the first stored
snapshot at or after it formed and in that snapshot's own disc frame:

  eps  = L_z / L_circ(E)   with L_circ from the potential's own R_circ(E)
  J_z  = vertical action, an adiabatic invariant

J_z is the quantity that separates a BAR orbit from a HALO orbit, which neither
circularity nor |z| can do.  Bar orbits are planar: low eps, but also low J_z.
Halo orbits are not: low eps AND high J_z.  |z| at birth confuses the two because
it depends on where in its orbit a star is caught, whereas J_z does not.

The potential is only fitted out to R = 50 kpc, so stars outside that get NaN
rather than an extrapolated action.

Writes out/birth_orbits_actions.npz.
"""
import gc, os
import numpy as np
import agama
import auriga_public as ap
import config_au18 as C

agama.setUnits(mass=1, length=1, velocity=1)
PDIR = C.OUT_DIR + '/potentials_ref'
idx = np.load(PDIR + '/index.npz')
POT_SNAP, POT_T, POT_AX = idx['snaps'], idx['t'], idx['axis']
RMAX_POT = 50.

st = np.load(C.OUT_DIR + '/snapshot_times.npz')
SNAPS, T_SNAP = st['snaps'], st['t_snap']
_cache = {}


def get_pot(t):
    k = int(POT_SNAP[np.argmin(np.abs(POT_T - t))])
    if k not in _cache:
        _cache.clear()
        p = agama.Potential(f'{PDIR}/pot_{k:03d}.ini')
        _cache[k] = (p, agama.ActionFinder(p))
    return (k,) + _cache[k]


def stars_in_frame(sn):
    """All star particles, centred and rotated into that snapshot's disc frame."""
    sub = ap.subhalos.subfind(sn, directory=C.SIM_DIR,
                              loadlist=['SubhaloPos', 'Group_R_Crit200'])
    r200 = float(sub.data['Group_R_Crit200'][0]); cen = sub.data['SubhaloPos'][0]
    ref = ap.snapshot.load_snapshot(sn, 4, snappath=C.SIM_DIR, verbose=False,
        loadlist=['Coordinates', 'Masses', 'Potential', 'Velocities'])
    ref = ap.util.CentreOnHalo(ref, cen)
    ref = ap.util.apply_mask(ref, stars=False, radialcut=.5 * r200)
    ist, = np.where(ap.util.r(ref) < .1 * r200)
    L = np.cross(ref.data['Coordinates'][ist],
                 ref.data['Velocities'][ist] * ref.data['Masses'][ist, None])
    Ld = L.sum(0); Ld /= np.sqrt((Ld ** 2).sum())
    xd, yd, zd = ap.util.get_principal_axis(ref, ist, L=Ld)
    del ref; gc.collect()

    # Reload WITHOUT the radial mask: the mask is right for fitting the potential
    # but would silently drop stars born beyond 0.5 R200, which at t = 2 Gyr is
    # only 26 kpc.
    s = ap.snapshot.load_snapshot(sn, 4, snappath=C.SIM_DIR, verbose=False,
        loadlist=['ParticleIDs', 'Coordinates', 'Masses', 'Velocities',
                  'GFM_StellarFormationTime'])
    s = ap.util.CentreOnHalo(s, cen)
    ap.util.rotateto(s, xd, dir2=yd, dir3=zd)
    real = s.data['GFM_StellarFormationTime'] > 0
    c = s.data['Coordinates'][real]; v = s.data['Velocities'][real]
    m = s.data['Masses'][real]; ids = s.data['ParticleIDs'][real]
    # rotateto puts the disc axis on component 0.  Map it to z with a CYCLIC
    # permutation (0,1,2) -> (1,2,0), determinant +1.  The (2,1,0) mapping used
    # by compute_auriga_potential.py is a transposition, determinant -1: a
    # reflection.  That is harmless when only a density is being fitted, but it
    # flips the sign of the angular momentum, which would put the disc at
    # eps = -1 instead of +1.
    pos = np.column_stack([c[:, 1], c[:, 2], c[:, 0]]) * 1e3     # -> (x, y, z), kpc
    vel = np.column_stack([v[:, 1], v[:, 2], v[:, 0]])
    rr = np.sqrt((pos * pos).sum(1)); inn = rr < 10.
    vel = vel - np.average(vel[inn], axis=0, weights=m[inn])
    del s, c, v; gc.collect()
    return ids, pos, vel, np.asarray(zd, float)


def measure(pos, vel, pot, af):
    R = np.hypot(pos[:, 0], pos[:, 1])
    Lz = pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]
    r = np.sqrt((pos * pos).sum(1))
    inside = r < RMAX_POT
    n = len(pos)
    eps = np.full(n, np.nan); Jr = np.full(n, np.nan)
    Jz = np.full(n, np.nan); Jp = np.full(n, np.nan); E = np.full(n, np.nan)
    if inside.any():
        w = np.column_stack([pos[inside], vel[inside]])
        phi = pot.potential(pos[inside])
        Ei = .5 * (vel[inside] ** 2).sum(1) + phi
        E[inside] = Ei
        bound = np.isfinite(Ei) & (Ei < 0)
        if bound.any():
            rc = pot.Rcirc(E=Ei[bound])
            good = np.isfinite(rc) & (rc > 0)
            ii = np.flatnonzero(inside)[np.flatnonzero(bound)[good]]
            rcg = rc[good]
            fr = pot.force(np.column_stack([rcg, 0 * rcg, 0 * rcg]))[:, 0]
            vc2 = -rcg * fr
            lc = np.where(vc2 > 0, rcg * np.sqrt(np.abs(vc2)), np.nan)
            eps[ii] = Lz[ii] / lc
        a = af(w)
        Jr[inside], Jz[inside], Jp[inside] = a[:, 0], a[:, 1], a[:, 2]
    return eps, Jr, Jz, Jp, E, Lz, R, pos[:, 2]


cat = np.load(C.OUT_DIR + '/z0_insitu_catalog.npz')
keep = cat['tform'] >= T_SNAP[0]
ids_k, tform_k = cat['ids'][keep], cat['tform'][keep]
assigned = np.searchsorted(T_SNAP, tform_k)
print(f'{len(ids_k):,} in-situ stars measurable near birth', flush=True)

KEYS = ('eps_birth', 'Jr_birth', 'Jz_birth', 'Jphi_birth', 'E_birth',
        'Lz_birth', 'R_birth', 'z_birth')
out = {k: np.full(len(ids_k), np.nan, np.float32) for k in KEYS}
pot_used = np.full(len(ids_k), -1, np.int16)
axis_now = np.full((len(SNAPS), 3), np.nan)

for k, sn in enumerate(SNAPS):
    sel = np.flatnonzero(assigned == k)
    if not len(sel): continue
    ids, pos, vel, axis_now[k] = stars_in_frame(int(sn))
    pk, pot, af = get_pot(float(T_SNAP[k]))
    res = measure(pos, vel, pot, af)
    o = np.argsort(ids); ss = ids[o]
    want = ids_k[sel]
    p = np.searchsorted(ss, want)
    ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == want)
    ix = o[p[ok]]; dst = sel[ok]
    for key, arr in zip(KEYS, res):
        out[key][dst] = arr[ix]
    pot_used[dst] = pk
    nb = np.isfinite(out['Jz_birth'][dst]).sum()
    print(f'snap {sn:3d} t={T_SNAP[k]:5.2f} pot {pk:3d}  born {len(sel):6,}  '
          f'matched {ok.sum():6,}  with Jz {nb:6,}', flush=True)
    del ids, pos, vel, res; gc.collect()

ids, pos, vel, axis_z0 = stars_in_frame(127)
pk, pot, af = get_pot(float(T_SNAP[-1]))
res = measure(pos, vel, pot, af)
o = np.argsort(ids); ss = ids[o]
p = np.searchsorted(ss, ids_k)
ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == ids_k)
ix = o[p[ok]]
Z0 = ('eps_z0', 'Jr_z0', 'Jz_z0', 'Jphi_z0', 'E_z0', 'Lz_z0', 'R_z0', 'z_z0')
z0 = {k: np.full(len(ids_k), np.nan, np.float32) for k in Z0}
for key, arr in zip(Z0, res):
    z0[key][ok] = arr[ix]
print(f'z=0 pass: matched {ok.sum():,}/{len(ids_k):,}', flush=True)

np.savez(C.OUT_DIR + '/birth_orbits_actions.npz', ids=ids_k, tform=tform_k,
         pot_used=pot_used, snaps=SNAPS, t_snap=T_SNAP,
         axis_now=axis_now, axis_z0=axis_z0, **out, **z0)
f = np.isfinite(out['Jz_birth'])
print(f"\nJ_z measured for {f.sum():,}/{len(ids_k):,} ({100 * f.mean():.1f}%)")
print('J_z percentiles [kpc km/s]:', np.round(np.percentile(out['Jz_birth'][f], [10, 50, 90, 99]), 1))
print('eps>1:', f"{100 * np.nanmean(out['eps_birth'] > 1):.2f}%")
print('saved', C.OUT_DIR + '/birth_orbits_actions.npz')
