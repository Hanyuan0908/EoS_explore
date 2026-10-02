"""z_max TODAY, by integrating orbits in the z = 0 AGAMA potential.

prep_zmax.py gets z_max at BIRTH from the vertical action, which is fast but
biased: fixing R at the guiding radius ignores that an eccentric orbit rises
higher where the disc is thinner, so it UNDERESTIMATES z_max by up to ~40 per
cent for eccentric orbits (METHOD_zmax_from_Jz.md).  Its own docstring says to
integrate the orbits if a real number is needed.  This does that, at z = 0.

    usage:  prep_zmax_z0.py [T_LO] [T_HI] [NMAX]      (default 4.5 6.5 0)

NMAX > 0 integrates only a random NMAX of the sample, for timing tests.

WHY ONE WOULD WANT IT.  Cutting the Eos sample on R or on instantaneous |z| to
remove the bar is unsatisfactory: R keeps bar stars currently at large radius,
and |z| is a snapshot of orbital phase, so it keeps a bar star caught at the top
of its small vertical oscillation and discards a halo star crossing the plane.
z_max is a property of the orbit, so a cut on it removes the bar as a dynamical
population rather than as a region or a phase.

STILL NOT SELECTION-NEUTRAL.  z_max correlates with being dynamically hot, so a
z_max cut will raise the born-hot fraction partly for reasons unconnected with
the bar.  It is cleaner than |z| but it is not free of that; the honest framing
is that R-cut and z_max-cut bracket the answer.

METHOD.  The potential is the CylSpline built by prep_potentials_ref.py for
snapshot 127.  It is AXISYMMETRIC, so the bar is averaged out of the orbit
integration -- which is exactly what makes a z_max cut usable as a bar-removal
tool, but also means these are not the true orbits of bar stars.  Initial
conditions come from z0_insitu_catalog (R, z, v_R, v_phi, v_z) with phi = 0,
which loses nothing in an axisymmetric potential.  Each orbit is integrated for
NPER circular periods and z_max is the maximum |z| along the trajectory.

THE UPPER TAIL IS NOT TO BE TRUSTED.  A few hundred stars are close enough to
unbound that the orbit simply runs away over 30 circular periods, and their
z_max is meaningless -- the maximum returned is 1.9e5 kpc.  Measured on the
4.5-6.5 Gyr sample: 0.61 per cent exceed 30 kpc, 0.27 per cent exceed 50, 0.11
per cent exceed 100 and 36 stars exceed 1000.  This is the same failure mode as
orbit_tools.apo_peri, and it does NOT affect a lower cut such as z_max > 2 kpc,
since those stars pass it either way.  Any analysis that uses z_max as a value
rather than as a threshold must clip the tail first.

Writes out/zmax_z0_<TLO>_<THI>.npz  (ids, zmax, rapo, rperi, nper, trajsize).
out/ is gitignored, so the file lives on this machine only; the script is what
is version-controlled, and a rerun takes ~15 min at ~500 orbits/s.
"""
import os, sys, time
import numpy as np
import agama
import config_au18 as C

T_LO = float(sys.argv[1]) if len(sys.argv) > 1 else 4.5
T_HI = float(sys.argv[2]) if len(sys.argv) > 2 else 6.5
NMAX = int(sys.argv[3]) if len(sys.argv) > 3 else 0
NPER, NT, CHUNK = 30, 800, 4000
R_MAX = 100.                      # beyond this the orbit is not meaningfully bound
OUT = C.OUT_DIR + f'/zmax_z0_{T_LO:g}_{T_HI:g}.npz'

agama.setUnits(mass=1, length=1, velocity=1)
pot = agama.Potential(C.OUT_DIR + '/potentials_ref/pot_127.ini')
print('potential: ' + C.OUT_DIR + '/potentials_ref/pot_127.ini')

cat = np.load(C.OUT_DIR + '/z0_insitu_catalog.npz')
sel = ((cat['tform'] >= T_LO) & (cat['tform'] < T_HI) & (cat['r'] < R_MAX)
       & np.isfinite(cat['vR']) & np.isfinite(cat['vphi']) & np.isfinite(cat['vz']))
idx = np.flatnonzero(sel)
if NMAX > 0 and NMAX < len(idx):
    idx = np.random.default_rng(0).choice(idx, NMAX, replace=False)
    print(f'TIMING RUN: {len(idx):,} of {int(sel.sum()):,}')
ids = cat['ids'][idx]
R = cat['R'][idx].astype(float)
z = cat['z'][idx].astype(float)
# phi = 0 without loss: the potential is axisymmetric
ic = np.column_stack([R, np.zeros_like(R), z,
                      cat['vR'][idx].astype(float),
                      cat['vphi'][idx].astype(float),
                      cat['vz'][idx].astype(float)])
print(f'{len(ic):,} orbits, {NPER} circular periods, trajsize {NT}')

Tc = pot.Tcirc(ic)
bad = ~np.isfinite(Tc) | (Tc <= 0)
Tc[bad] = np.nanmedian(Tc[~bad])
zmax = np.full(len(ic), np.nan)
rapo = np.full(len(ic), np.nan)
rperi = np.full(len(ic), np.nan)
t0 = time.time()
for a in range(0, len(ic), CHUNK):
    b = min(a + CHUNK, len(ic))
    orb = agama.orbit(potential=pot, ic=ic[a:b], time=NPER * Tc[a:b], trajsize=NT)
    for i in range(b - a):
        xv = orb[i][1]
        zmax[a + i] = np.abs(xv[:, 2]).max()
        rr = np.sqrt((xv[:, :3] ** 2).sum(1))
        rapo[a + i] = rr.max()
        rperi[a + i] = rr.min()
    el = time.time() - t0
    print(f'  {b:>8,} / {len(ic):,}   {el:7.1f} s   '
          f'({b / max(el, 1e-9):.0f} orbits/s, eta {el * (len(ic) - b) / b / 60:5.1f} min)',
          flush=True)
    del orb

ok = np.isfinite(zmax)
print(f'\nz_max: {ok.sum():,} integrated')
for q in (5, 16, 50, 84, 95, 99):
    print(f'  {q:>2d}th percentile  {np.nanpercentile(zmax, q):7.2f} kpc')
if NMAX == 0:
    np.savez(OUT, ids=ids, zmax=zmax, rapo=rapo, rperi=rperi,
             nper=NPER, trajsize=NT)
    print(f'\nsaved {OUT}')
else:
    print('\ntiming run: nothing written')
