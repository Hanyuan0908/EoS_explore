"""Birth orbit of every in-situ Au18 star: was it born on a disc or a halo orbit?

TWO circularities are stored per star, because the one the rest of this project
uses is not actually jz/jcirc(E):

  eps_env  jz divided by the 95th percentile of jz among prograde stars in the
           same energy bin.  This is the estimator in ana_z0_kinematic_catalog.py
           and ana_premerger_splash.py, kept here so earlier results remain
           reproducible.  It is an empirical envelope, and by construction ~5 per
           cent of prograde stars exceed it, so eps_env > 1 is routine and the
           normalisation drifts with the shape of the prograde distribution --
           which changes a lot between a disordered z=2 galaxy and the z=0 disc.
  eps_sph  jz/jcirc(E) with jcirc computed properly from the spherically averaged
           potential: v_c^2 = r dPhi/dr, E_c(r) = Phi + v_c^2/2, and jcirc = r v_c
           inverted onto E.  Bounded by ~1 by construction and, more importantly,
           normalised by the potential rather than by the star distribution, so it
           can be compared between snapshots.

The disc angular-momentum axis is also recorded per snapshot (before alignment),
because eps is measured against the CONTEMPORARY disc plane at birth and against
the z=0 plane today; if the disc tilts between the two, stars change eps without
changing orbit.

For each snapshot from SNAP_MIN to SNAP_MAX this measures the circularity
eps = jz/jcirc(E), the azimuthal velocity and the birth position of the stars
that formed since the previous stored snapshot, i.e. every star is caught in the
first snapshot at or after it formed.  Snapshot spacing is ~0.15 Gyr, so "birth"
means "within ~0.15 Gyr of birth" -- long enough for a star to move along its
orbit but far too short for secular heating to change which orbit it is on.

The circularity estimator is copied verbatim from ana_premerger_splash.py so that
eps here is on the same scale as the A/B/C channel definitions.

Stars formed before the first stored snapshot cannot be caught near birth and are
left out; they are ~7 per cent of the in-situ mass.

Writes out/birth_orbits.npz.  Run time ~10 min.
"""
import gc, os, sys
import numpy as np
import config_au18 as C
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import orbit_tools as OT
from auriga_public import snapshot as snap_mod, subhalos as sub_mod, util

os.makedirs(C.OUT_DIR, exist_ok=True)
SNAPS = np.arange(C.SNAP_MIN, C.SNAP_MAX + 1)


def aligned_snapshot(sn):
    s = snap_mod.load_snapshot(sn, 4, snappath=C.SIM_DIR,
        loadlist=['ParticleIDs', 'Coordinates', 'Velocities', 'Masses',
                  'Potential', 'GFM_StellarFormationTime'])
    real = s.data['GFM_StellarFormationTime'] > 0        # wind particles are < 0
    for key in list(s.data): s.data[key] = s.data[key][real]
    sf = sub_mod.subfind(sn, directory=C.SIM_DIR, loadlist=['GroupFirstSub', 'SubhaloPos'])
    cen = sf.data['SubhaloPos'][int(sf.data['GroupFirstSub'][0])]
    util.CentreOnHalo(s, cen)
    r0 = np.sqrt((s.data['Coordinates'] ** 2).sum(1)); inner = r0 < .01
    bulk = np.average(s.data['Velocities'][inner], axis=0, weights=s.data['Masses'][inner])
    # The disc axis in the centred but unrotated frame, for the tilt diagnostic.
    x0 = s.data['Coordinates'] * 1000.; v0 = s.data['Velocities']
    r0k = np.sqrt((x0 * x0).sum(1)); inn = r0k < 10.
    J = (s.data['Masses'][inn, None] * np.cross(x0[inn], v0[inn])).sum(0)
    axis = J / np.linalg.norm(J)
    s.data['Velocities'] -= bulk; util.align_galaxy(s, radialcut=.01)
    return s, axis


def jcirc_of_E(rc, prof, k_out, E):
    """jcirc(E) from the spherically averaged potential, inverted onto energy."""
    rg = np.logspace(np.log10(max(rc[0], .05)), np.log10(min(rc[-1], 500.)), 800)
    ph = OT.phi_interp(rg, rc, prof, k_out)
    vc2 = rg * np.gradient(ph, rg)
    ok = np.isfinite(vc2) & (vc2 > 0)
    rg, ph, vc2 = rg[ok], ph[ok], vc2[ok]
    Ec, jc = ph + .5 * vc2, rg * np.sqrt(vc2)
    keep = np.concatenate([[True], np.diff(np.maximum.accumulate(Ec)) > 0])
    Ec, jc = Ec[keep], jc[keep]
    return np.interp(E, Ec, jc, left=jc[0], right=jc[-1])


def kinematics(s):
    """Both circularities, cylindrical velocities and position."""
    x = s.data['Coordinates'] * 1000.; v = s.data['Velocities']
    r = np.sqrt((x * x).sum(1))
    R = np.hypot(x[:, 1], x[:, 2]); jz = x[:, 1] * v[:, 2] - x[:, 2] * v[:, 1]
    disc = (R > 3) & (R < 12) & (np.abs(x[:, 0]) < 2)
    flip = -1. if (disc.sum() > 20 and np.median(jz[disc]) < 0) else 1.
    jz = jz * flip
    E = .5 * (v * v).sum(1) + s.data['Potential']
    valid = np.isfinite(E) & np.isfinite(jz) & (r < 50)
    edges = np.quantile(E[valid], np.linspace(0, 1, 241))
    ib = np.clip(np.searchsorted(edges, E, 'right') - 1, 0, 239)
    jc = np.full(240, np.nan)
    for b in range(240):
        q = valid & (ib == b) & (jz > 0)
        if q.sum() > 30: jc[b] = np.percentile(jz[q], 95)
    ok = np.isfinite(jc)
    jc = np.interp(np.arange(240), np.flatnonzero(ok), jc[ok])
    eps_env = jz / jc[ib]

    # Proper jcirc(E).  E must be the SPHERICAL energy to match the profile,
    # otherwise the flattening of the disc pushes stars below the Phi_eff floor.
    rc, prof, k_out = OT.potential_profile(r, s.data['Potential'].astype(np.float64))
    E_sph = OT.spherical_energy((v * v).sum(1), r, rc, prof, k_out)
    eps_sph = jz / jcirc_of_E(rc, prof, k_out, E_sph)

    Rs = np.where(R > .1, R, 1.)
    vR = (x[:, 1] * v[:, 1] + x[:, 2] * v[:, 2]) / Rs
    vphi = flip * (x[:, 1] * v[:, 2] - x[:, 2] * v[:, 1]) / Rs
    return eps_env, eps_sph, jz, E_sph, vR, vphi, R, x[:, 0], r


# --- which snapshot catches each star just after it formed -------------------
cat = np.load(C.OUT_DIR + '/z0_insitu_catalog.npz')
ids, tform = cat['ids'], cat['tform']

apath = C.OUT_DIR + '/snapshot_times.npz'
if os.path.exists(apath):
    a_snap = np.load(apath)['a_snap']
else:
    a_snap = np.array([float(snap_mod.load_snapshot(
        int(sn), 4, snappath=C.SIM_DIR, loadlist=['GFM_StellarFormationTime']).time)
        for sn in SNAPS])
    np.savez(apath, snaps=SNAPS, a_snap=a_snap, t_snap=C.a_to_age(a_snap))
t_snap = C.a_to_age(a_snap)
print(f'snapshots {SNAPS[0]}-{SNAPS[-1]}: t = {t_snap[0]:.2f}-{t_snap[-1]:.2f} Gyr, '
      f'median spacing {np.median(np.diff(t_snap)) * 1000:.0f} Myr', flush=True)

keep = tform >= t_snap[0]
assigned = np.searchsorted(t_snap, tform[keep])          # first snap at or after birth
ids_k, tform_k = ids[keep], tform[keep]
print(f'in-situ stars {len(ids):,}; measurable near birth {keep.sum():,} '
      f'({100 * keep.mean():.1f} per cent)', flush=True)

KEYS = ('eps_birth', 'eps_sph_birth', 'jz_birth', 'E_sph_birth', 'vR_birth',
        'vphi_birth', 'R_birth', 'z_birth', 'r_birth')
out = {k: np.full(len(ids_k), np.nan, np.float32) for k in KEYS}
snap_birth = np.full(len(ids_k), -1, np.int16)
axes = np.full((len(SNAPS), 3), np.nan)

for k, sn in enumerate(SNAPS):
    sel = np.flatnonzero(assigned == k)
    if not len(sel): continue
    s, axes[k] = aligned_snapshot(int(sn))
    eps, eps_s, jz, Es, vR, vphi, R, zc, r = kinematics(s)
    sid = s.data['ParticleIDs']; o = np.argsort(sid); ss = sid[o]
    want = ids_k[sel]
    p = np.searchsorted(ss, want)
    ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == want)
    ix = o[p[ok]]; dst = sel[ok]
    for key, arr in [('eps_birth', eps), ('eps_sph_birth', eps_s), ('jz_birth', jz),
                     ('E_sph_birth', Es), ('vR_birth', vR), ('vphi_birth', vphi),
                     ('R_birth', R), ('z_birth', zc), ('r_birth', r)]:
        out[key][dst] = arr[ix]
    snap_birth[dst] = sn
    print(f'snap {sn:3d}  t={t_snap[k]:5.2f}  born {len(sel):7,}  recovered {ok.sum():7,}',
          flush=True)
    del s, eps, eps_s, jz, Es, vR, vphi, R, zc, r, sid, o, ss; gc.collect()

# z=0 on the same footing, for every star, so birth and present use one estimator.
s, axis_z0 = aligned_snapshot(127)
eps, eps_s, jz, Es, vR, vphi, R, zc, r = kinematics(s)
sid = s.data['ParticleIDs']; o = np.argsort(sid); ss = sid[o]
p = np.searchsorted(ss, ids_k)
okz = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == ids_k)
ixz = o[p[okz]]
z0 = {k: np.full(len(ids_k), np.nan, np.float32)
      for k in ('eps_z0', 'eps_sph_z0', 'vR_z0', 'vphi_z0', 'R_z0', 'r_z0')}
for key, arr in [('eps_z0', eps), ('eps_sph_z0', eps_s), ('vR_z0', vR),
                 ('vphi_z0', vphi), ('R_z0', R), ('r_z0', r)]:
    z0[key][okz] = arr[ixz]
print(f'z=0 pass: matched {okz.sum():,}/{len(ids_k):,}', flush=True)

tilt = np.degrees(np.arccos(np.clip(axes @ axis_z0, -1, 1)))
print('\ndisc-axis tilt relative to z=0:')
for k, sn in enumerate(SNAPS):
    if np.isfinite(tilt[k]) and sn % 4 == 0:
        print(f'  snap {sn:3d}  t={t_snap[k]:5.2f}  tilt = {tilt[k]:5.1f} deg', flush=True)

np.savez(C.OUT_DIR + '/birth_orbits.npz', ids=ids_k, tform=tform_k,
         snap_birth=snap_birth, snaps=SNAPS, t_snap=t_snap,
         disc_axis=axes, disc_axis_z0=axis_z0, tilt_deg=tilt, **out, **z0)
n = np.isfinite(out['eps_birth']).sum()
print(f'\nmeasured {n:,}/{len(ids_k):,}; saved {C.OUT_DIR}/birth_orbits.npz')
