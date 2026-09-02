"""Axisymmetric AGAMA CylSpline potentials for Au18, one every ~0.5 Gyr.

Built from the actual particle distribution -- gas, high-res DM, both low-res DM
boundary types and stars within 500 kpc of the main subhalo -- rather than from a
spherical average of the stored per-particle potential.  Each potential is built
in a frame centred on the main subhalo and rotated so that +z is the disc angular
momentum axis AT THAT SNAPSHOT, then exported as an AGAMA .ini.

Sampling is 0.5 Gyr as a baseline, but refined to every snapshot between 4.5 and
7.0 Gyr.  The reason is not the mass distribution, which evolves slowly, but the
ORIENTATION: the disc axis swings 54 degrees between t = 4.99 and t = 5.59 Gyr as
the GS/E merger comes in, so a 0.5 Gyr grid would hand a star a potential whose
symmetry axis is tens of degrees away from its own disc plane.

Writes out/potentials/pot_<snap>.ini plus out/potentials/index.npz.
"""
import gc, os
import numpy as np
import agama
import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod

agama.setUnits(mass=1, length=1, velocity=1)          # Msun, kpc, km/s
PDIR = C.OUT_DIR + '/potentials'
os.makedirs(PDIR, exist_ok=True)

RMAX_PART = 500.                                       # particles used, kpc
REFINE = (4.5, 7.0)                                    # every snapshot in here
PTYPES = (0, 1, 2, 3, 4)

st = np.load(C.OUT_DIR + '/snapshot_times.npz')
snaps, t_snap = st['snaps'], st['t_snap']

want = []
for t in np.arange(np.ceil(t_snap[0] * 2) / 2, t_snap[-1] + .01, .5):
    want.append(int(snaps[np.argmin(np.abs(t_snap - t))]))
want += [int(s) for s, t in zip(snaps, t_snap) if REFINE[0] <= t <= REFINE[1]]
want = sorted(set(want + [int(snaps[-1])]))
print(f'{len(want)} potentials: snaps {want[0]}-{want[-1]}', flush=True)


def disc_axis(sn, cen):
    """Angular momentum direction of the stars inside 10 kpc, unrotated frame."""
    s = snap_mod.load_snapshot(sn, 4, snappath=C.SIM_DIR,
        loadlist=['Coordinates', 'Velocities', 'Masses', 'GFM_StellarFormationTime'])
    real = s.data['GFM_StellarFormationTime'] > 0
    x = (s.data['Coordinates'][real] - cen) * 1000.
    v = s.data['Velocities'][real]; m = s.data['Masses'][real]
    r = np.sqrt((x * x).sum(1)); inn = r < 10.
    # The bulk velocity has to come out before the angular momentum is meaningful.
    v = v - np.average(v[inn], axis=0, weights=m[inn])
    J = (m[inn, None] * np.cross(x[inn], v[inn])).sum(0)
    del s; gc.collect()
    return J / np.linalg.norm(J)


def rotation_to(axis):
    """Rotation matrix sending `axis` to +z, with an arbitrary but fixed azimuth."""
    z = axis / np.linalg.norm(axis)
    tmp = np.array([1., 0., 0.])
    if abs(np.dot(tmp, z)) > .9: tmp = np.array([0., 1., 0.])
    x = np.cross(tmp, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.vstack([x, y, z])                        # rows are the new basis


rows = []
for sn in want:
    f = f'{PDIR}/pot_{sn:03d}.ini'
    sf = sub_mod.subfind(sn, directory=C.SIM_DIR, loadlist=['GroupFirstSub', 'SubhaloPos'])
    cen = sf.data['SubhaloPos'][int(sf.data['GroupFirstSub'][0])]
    ax = disc_axis(sn, cen)
    rows.append((sn, cen, ax))
    if os.path.exists(f):
        print(f'snap {sn:3d}  potential exists, skipped', flush=True); continue
    R = rotation_to(ax)
    P, M = [], []
    for pt in PTYPES:
        s = snap_mod.load_snapshot(sn, pt, snappath=C.SIM_DIR,
                                   loadlist=['Coordinates', 'Masses'])
        x = (s.data['Coordinates'] - cen) * 1000.
        m = s.data['Masses'] * C.MASS_TO_MSUN
        q = (x * x).sum(1) < RMAX_PART ** 2
        P.append(x[q] @ R.T); M.append(m[q])
        del s, x, m, q; gc.collect()
    P = np.concatenate(P); M = np.concatenate(M)
    pot = agama.Potential(type='CylSpline', particles=(P, M), symmetry='axisymmetric',
                          gridsizeR=30, gridsizez=30, Rmin=.1, Rmax=400.,
                          zmin=.05, zmax=400., mmax=0)
    pot.export(f)
    vc = np.sqrt(-8. * pot.force([[8., 0., 0.]])[0][0])
    print(f'snap {sn:3d}  N={len(P):>9,}  M={M.sum():.3e}  v_c(8kpc)={vc:6.1f} km/s  -> {f}',
          flush=True)
    del P, M, pot; gc.collect()

np.savez(PDIR + '/index.npz',
         snaps=np.array([r[0] for r in rows]),
         t=np.array([float(t_snap[list(snaps).index(r[0])]) for r in rows]),
         centre=np.array([r[1] for r in rows]),
         axis=np.array([r[2] for r in rows]))
print(f'\nwrote {PDIR}/index.npz with {len(rows)} entries')
