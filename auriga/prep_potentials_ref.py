"""AGAMA CylSpline potentials for Au18, following compute_auriga_potential.py.

Replaces prep_potentials_agama.py, whose potentials were unusable for actions:
they included the low-res boundary DM (types 2 and 3) and ran the grid out to
400 kpc, which left so little resolution in the centre that the circular-orbit
energy was non-monotonic inside ~0.5 kpc (v_c^2 < 0, an outward force).  AGAMA's
ActionFinder refuses to initialise on such a potential.

The recipe here is the reference one: particle types 4, 1 and 0 (stars, high-res
DM, gas) inside 0.5 R200, aligned by the inertia-tensor principal axis closest to
the stellar angular momentum, CylSpline with gridSizeR=30, gridSizez=25,
Rmin=0.15, Rmax=50, zmin=0.1, zmax=20.  That gives a monotonic rotation curve
(v_c = 245 km/s at 8 kpc at t = 6.2 Gyr) and a working ActionFinder.

Note Rmax = 50 kpc: the potential is only trustworthy inside that, which covers
every birth radius in this analysis but not the far halo.

Same sampling as before -- 0.5 Gyr, refined to every snapshot in 4.5-7.0 Gyr,
because the disc axis swings 54 degrees across coalescence.
Writes out/potentials_ref/pot_<snap>.ini and index.npz.
"""
import gc, os
import numpy as np
import agama
import auriga_public as ap
import config_au18 as C

agama.setUnits(mass=1, length=1, velocity=1)
PDIR = C.OUT_DIR + '/potentials_ref'
os.makedirs(PDIR, exist_ok=True)
REFINE = (4.5, 7.0)

st = np.load(C.OUT_DIR + '/snapshot_times.npz')
snaps, t_snap = st['snaps'], st['t_snap']
want = [int(snaps[np.argmin(np.abs(t_snap - t))])
        for t in np.arange(np.ceil(t_snap[0] * 2) / 2, t_snap[-1] + .01, .5)]
want += [int(s) for s, t in zip(snaps, t_snap) if REFINE[0] <= t <= REFINE[1]]
want = sorted(set(want + [int(snaps[-1])]))
print(f'{len(want)} potentials: snaps {want[0]}-{want[-1]}', flush=True)


def frame(sn):
    """Centre, principal-axis alignment, and the disc axis, from the stars."""
    sub = ap.subhalos.subfind(sn, directory=C.SIM_DIR,
                              loadlist=['SubhaloPos', 'Group_R_Crit200'])
    r200 = float(sub.data['Group_R_Crit200'][0]); cen = sub.data['SubhaloPos'][0]
    s = ap.snapshot.load_snapshot(sn, 4, snappath=C.SIM_DIR, verbose=False,
        loadlist=['Coordinates', 'Masses', 'Potential', 'Velocities'])
    s = ap.util.CentreOnHalo(s, cen)
    s = ap.util.apply_mask(s, stars=False, radialcut=.5 * r200)
    ist, = np.where(ap.util.r(s) < .1 * r200)
    L = np.cross(s.data['Coordinates'][ist], s.data['Velocities'][ist] * s.data['Masses'][ist, None])
    Ld = L.sum(0); Ld /= np.sqrt((Ld ** 2).sum())
    xd, yd, zd = ap.util.get_principal_axis(s, ist, L=Ld)
    return sub, r200, cen, xd, yd, zd, Ld, s


rows = []
for sn in want:
    f = f'{PDIR}/pot_{sn:03d}.ini'
    sub, r200, cen, xd, yd, zd, Ld, s = frame(int(sn))
    rows.append((sn, cen, np.asarray(zd, float), Ld))
    if os.path.exists(f):
        print(f'snap {sn:3d}  exists, skipped', flush=True); del s; gc.collect(); continue
    ap.util.rotateto(s, xd, dir2=yd, dir3=zd)
    X, Y, Z, M = [], [], [], []

    def add(o):
        X.append(o.data['Coordinates'][:, 2] * 1e3); Y.append(o.data['Coordinates'][:, 1] * 1e3)
        Z.append(o.data['Coordinates'][:, 0] * 1e3); M.append(o.data['Masses'])

    add(s); del s; gc.collect()
    for pt in (1, 0):
        o = ap.snapshot.load_snapshot(int(sn), pt, snappath=C.SIM_DIR, verbose=False,
            loadlist=['Coordinates', 'Masses', 'Potential', 'Velocities'])
        o = ap.util.CentreOnHalo(o, cen)
        o = ap.util.apply_mask(o, stars=False, radialcut=.5 * r200)
        ap.util.rotateto(o, xd, dir2=yd, dir3=zd)
        add(o); del o; gc.collect()
    X, Y, Z, M = np.hstack(X), np.hstack(Y), np.hstack(Z), np.hstack(M)
    q = np.sqrt(X * X + Y * Y + Z * Z) > 0
    pot = agama.Potential(type='CylSpline', particles=(np.column_stack([X, Y, Z])[q], M[q] * 1e10),
                          symmetry='axisymmetric', gridSizeR=30, gridSizez=25,
                          Rmin=0.15, Rmax=50, zmin=0.1, zmax=20)
    pot.export(f)
    vc = np.sqrt(-8. * pot.force([[8., 0., 0.]])[0][0])
    try:
        agama.ActionFinder(pot); af = 'actions OK'
    except Exception as e:
        af = 'ACTIONS FAIL'
    print(f'snap {sn:3d}  N={q.sum():>9,}  R200={r200 * 1e3:6.1f}  v_c(8)={vc:6.1f}  {af}', flush=True)
    del X, Y, Z, M, pot; gc.collect()

np.savez(PDIR + '/index.npz',
         snaps=np.array([r[0] for r in rows]),
         t=np.array([float(t_snap[list(snaps).index(r[0])]) for r in rows]),
         centre=np.array([r[1] for r in rows]),
         axis=np.array([r[2] for r in rows]),
         Ldir=np.array([r[3] for r in rows]))
print(f'\nwrote {PDIR}/index.npz with {len(rows)} entries')
