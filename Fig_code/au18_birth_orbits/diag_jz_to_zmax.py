"""Convert J_z into a maximum vertical excursion, and check it against orbits.

z_max is recovered by inverting the vertical action at the guiding radius,

    J_z = (1/pi) Int_0^zmax sqrt( 2 [ Phi(Rg, zmax) - Phi(Rg, z) ] ) dz ,

which is monotonic in zmax.  A table of J_z(Rg, zmax) is built once per potential
and inverted by interpolation.  The substitution z = zmax sin(u) removes the
square-root singularity at the turning point.

The check is the honest one: integrate the same stars' orbits in the same static
potential and compare max|z| along the trajectory with the action-derived value.
The epicyclic estimate sqrt(2 Jz/nu) is shown alongside, since it is the small
amplitude limit and is expected to fail for the orbits we actually care about.
"""
import sys
import numpy as np
import agama
import auriga_public as ap
import config_au18 as C

agama.setUnits(mass=1, length=1, velocity=1)
SNAP = int(sys.argv[1]) if len(sys.argv) > 1 else 80
NTEST = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
pot = agama.Potential(f'{C.OUT_DIR}/potentials_ref/pot_{SNAP:03d}.ini')
af = agama.ActionFinder(pot)

# ---- table J_z(Rg, zmax) -----------------------------------------------------
NG, NZ, NQ = 48, 64, 48
Rg_grid = np.logspace(np.log10(.3), np.log10(45.), NG)
zm_grid = np.logspace(np.log10(.02), np.log10(30.), NZ)
u, w = np.polynomial.legendre.leggauss(NQ)
u = .5 * np.pi * .5 * (u + 1); w = .5 * np.pi * .5 * w        # u in [0, pi/2]


def jz_table():
    T = np.zeros((NG, NZ))
    for i, Rg in enumerate(Rg_grid):
        top = pot.potential(np.column_stack([np.full(NZ, Rg), np.zeros(NZ), zm_grid]))
        for j, zm in enumerate(zm_grid):
            z = zm * np.sin(u)
            ph = pot.potential(np.column_stack([np.full(NQ, Rg), np.zeros(NQ), z]))
            val = 2. * (top[j] - ph)
            # J_z = (1/2pi) * closed loop integral of v_z dz.  The loop is
            # -zmax -> +zmax -> -zmax, so it is 4x the quarter integral from 0 to
            # zmax, giving the (2/pi) prefactor -- not (1/pi).
            T[i, j] = 2. * np.sum(w * np.sqrt(np.maximum(val, 0)) * zm * np.cos(u)) / np.pi
    return T


TAB = jz_table()
print(f'J_z table built: Rg {Rg_grid[0]:.2f}-{Rg_grid[-1]:.1f} kpc, '
      f'zmax {zm_grid[0]:.2f}-{zm_grid[-1]:.1f} kpc')


def zmax_from_Jz(Jz, Rg):
    """Invert the table; monotonic in zmax so np.interp per Rg row suffices."""
    i = np.clip(np.searchsorted(Rg_grid, Rg) - 1, 0, NG - 2)
    f = (Rg - Rg_grid[i]) / (Rg_grid[i + 1] - Rg_grid[i])
    out = np.full(len(Jz), np.nan)
    for k in range(len(Jz)):
        row = (1 - f[k]) * TAB[i[k]] + f[k] * TAB[i[k] + 1]
        if Jz[k] <= row[0]:
            out[k] = zm_grid[0] * np.sqrt(max(Jz[k], 0) / max(row[0], 1e-30))
        elif Jz[k] >= row[-1]:
            out[k] = zm_grid[-1]
        else:
            out[k] = np.interp(Jz[k], row, zm_grid)
    return out


# ---- stars ------------------------------------------------------------------
sub = ap.subhalos.subfind(SNAP, directory=C.SIM_DIR,
                          loadlist=['SubhaloPos', 'Group_R_Crit200'])
r200 = float(sub.data['Group_R_Crit200'][0]); cen = sub.data['SubhaloPos'][0]
ref = ap.snapshot.load_snapshot(SNAP, 4, snappath=C.SIM_DIR, verbose=False,
    loadlist=['Coordinates', 'Masses', 'Potential', 'Velocities'])
ref = ap.util.CentreOnHalo(ref, cen)
ref = ap.util.apply_mask(ref, stars=False, radialcut=.5 * r200)
ist, = np.where(ap.util.r(ref) < .1 * r200)
L = np.cross(ref.data['Coordinates'][ist],
             ref.data['Velocities'][ist] * ref.data['Masses'][ist, None])
Ld = L.sum(0); Ld /= np.sqrt((Ld ** 2).sum())
xd, yd, zd = ap.util.get_principal_axis(ref, ist, L=Ld)
ap.util.rotateto(ref, xd, dir2=yd, dir3=zd)
c = ref.data['Coordinates']; v = ref.data['Velocities']; m = ref.data['Masses']
pos = np.column_stack([c[:, 2], c[:, 1], c[:, 0]]) * 1e3
vel = np.column_stack([v[:, 2], v[:, 1], v[:, 0]])
rr = np.sqrt((pos * pos).sum(1)); inn = rr < 10.
vel = vel - np.average(vel[inn], axis=0, weights=m[inn])

sel = np.flatnonzero(rr < 40.)
rng = np.random.default_rng(1)
sel = rng.choice(sel, size=min(NTEST, len(sel)), replace=False)
w6 = np.column_stack([pos[sel], vel[sel]])
acts = af(w6)
Jr, Jz, Jphi = acts[:, 0], acts[:, 1], acts[:, 2]
Rg = pot.Rcirc(L=np.abs(Jphi))
good = np.isfinite(Jz) & np.isfinite(Rg) & (Rg > Rg_grid[0]) & (Rg < Rg_grid[-1]) & (Jz > 0)
print(f'{good.sum()}/{len(sel)} test stars usable')

zmax_act = zmax_from_Jz(Jz[good], Rg[good])

# epicyclic comparison: nu^2 = d2Phi/dz2 at (Rg, 0)
h = 1e-3
P = lambda R, z: pot.potential(np.column_stack([R, np.zeros_like(R), np.full_like(R, z)]))
Rgg = Rg[good]
nu2 = (P(Rgg, h) - 2 * P(Rgg, 0.) + P(Rgg, -h)) / h ** 2
zmax_epi = np.sqrt(2 * Jz[good] / np.sqrt(np.maximum(nu2, 1e-30)))

# ---- ground truth: integrate the orbits -------------------------------------
ic = w6[good]
Tc = pot.Tcirc(ic)
NT = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
orb = agama.orbit(potential=pot, ic=ic, time=40 * Tc, trajsize=NT)
zmax_orb = np.array([np.max(np.abs(o[1][:, 2])) for o in orb])
print(f'integrated {len(ic)} orbits, 40 circular periods, trajsize={NT} '
      f'({NT / 40:.0f} samples per circular period)')

for lab, est in [('action inversion', zmax_act), ('epicyclic sqrt(2Jz/nu)', zmax_epi)]:
    r = est / zmax_orb
    print(f'\n{lab}:  median est/true = {np.median(r):.3f}  '
          f'16-84 pct = {np.percentile(r, 16):.3f}-{np.percentile(r, 84):.3f}')
    for lo, hi in [(0, .5), (.5, 2), (2, 5), (5, 50)]:
        q = (zmax_orb >= lo) & (zmax_orb < hi)
        if q.sum() > 20:
            print(f'   true zmax {lo:4.1f}-{hi:<4.1f} kpc  N={q.sum():5d}  '
                  f'median est/true = {np.median(r[q]):.3f}')
np.savez(C.OUT_DIR + f'/jz_zmax_check_snap{SNAP}.npz', Jz=Jz[good], Rg=Rg[good],
         zmax_act=zmax_act, zmax_epi=zmax_epi, zmax_orb=zmax_orb)
print('\nsaved check to', C.OUT_DIR + f'/jz_zmax_check_snap{SNAP}.npz')
