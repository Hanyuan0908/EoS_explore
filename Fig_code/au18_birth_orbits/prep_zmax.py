"""Approximate maximum vertical excursion z_max from the birth vertical action.

z_max is recovered by inverting the vertical action at the guiding radius,

    J_z = (2/pi) Int_0^zmax sqrt( 2 [ Phi(Rg, zmax) - Phi(Rg, z) ] ) dz ,

which is monotonic in zmax, in each star's own birth potential (potentials_ref).
Rg = pot.Rcirc(L=|J_phi|).  The (2/pi) prefactor is the one that converges to the
true value in the near-circular limit; (1/pi) is wrong by 37 per cent there.

ACCURACY, measured against direct orbit integration at snapshot 80
(diag_jz_to_zmax.py, 1422 orbits):

    J_r/(J_r+|J_phi|)     median z_max(action) / z_max(orbit)
      < 0.02  circular              0.92
      0.02-0.05                     0.85
      0.05-0.15                     0.77
      > 0.15  eccentric             0.57

So this UNDERESTIMATES z_max, mildly for circular orbits and by up to ~40 per
cent for eccentric ones, because fixing R at Rg ignores that an eccentric orbit
climbs higher near apocentre where the vertical restoring force is weakest.  It
is used here as a classifier, not as a measurement: it is monotonic in J_z at
fixed Rg, so it ranks orbits correctly and only the effective threshold shifts.
A cut at z_max > 2 kpc therefore behaves like a somewhat stricter cut on the true
z_max, more so for radially eccentric stars.  Do not quote these z_max as physical
heights; integrate the orbits if a real number is needed.

Writes z_max into out/birth_orbits_zmax.npz.
"""
import numpy as np
import agama
import config_au18 as C

agama.setUnits(mass=1, length=1, velocity=1)
PDIR = C.OUT_DIR + '/potentials_ref'
NG, NZ, NQ, NJ = 56, 80, 48, 200
Rg_grid = np.logspace(np.log10(.3), np.log10(48.), NG)
zm_grid = np.logspace(np.log10(.02), np.log10(40.), NZ)
u, w = np.polynomial.legendre.leggauss(NQ)
u = .25 * np.pi * (u + 1); w = .25 * np.pi * w          # u in [0, pi/2]


def zmax_table(pot):
    """z_max on a regular (log Rg, log Jz) grid, by inverting J_z(Rg, zmax)."""
    T = np.zeros((NG, NZ))
    for i, Rg in enumerate(Rg_grid):
        top = pot.potential(np.column_stack([np.full(NZ, Rg), np.zeros(NZ), zm_grid]))
        for j, zm in enumerate(zm_grid):
            z = zm * np.sin(u)
            ph = pot.potential(np.column_stack([np.full(NQ, Rg), np.zeros(NQ), z]))
            T[i, j] = 2. * np.sum(w * np.sqrt(np.maximum(2. * (top[j] - ph), 0))
                                  * zm * np.cos(u)) / np.pi
    Jz_grid = np.logspace(np.log10(max(T[:, 0].min(), 1e-4)), np.log10(T[:, -1].max()), NJ)
    Z = np.empty((NG, NJ))
    for i in range(NG):
        lt = np.log(np.maximum(T[i], 1e-30))
        Z[i] = np.exp(np.interp(np.log(Jz_grid), lt, np.log(zm_grid)))
        # below the table, vertical motion is harmonic: J_z ~ z^2
        low = Jz_grid < T[i, 0]
        Z[i][low] = zm_grid[0] * np.sqrt(Jz_grid[low] / T[i, 0])
    return Jz_grid, Z


def zmax_of(pot, Jz, Rg):
    Jg, Z = zmax_table(pot)
    li, lj = np.log(Rg_grid), np.log(Jg)
    x = np.clip(np.log(Rg), li[0], li[-1]); y = np.clip(np.log(Jz), lj[0], lj[-1])
    i = np.clip(np.searchsorted(li, x) - 1, 0, NG - 2)
    j = np.clip(np.searchsorted(lj, y) - 1, 0, NJ - 2)
    fx = (x - li[i]) / (li[i + 1] - li[i]); fy = (y - lj[j]) / (lj[j + 1] - lj[j])
    lZ = np.log(Z)
    v = ((1 - fx) * (1 - fy) * lZ[i, j] + fx * (1 - fy) * lZ[i + 1, j]
         + (1 - fx) * fy * lZ[i, j + 1] + fx * fy * lZ[i + 1, j + 1])
    return np.exp(v)


a = np.load(C.OUT_DIR + '/birth_orbits_actions.npz')
n = len(a['ids'])
zmax_b = np.full(n, np.nan, np.float32)
Rg_b = np.full(n, np.nan, np.float32)
for pk in np.unique(a['pot_used']):
    if pk < 0: continue
    m = (a['pot_used'] == pk) & np.isfinite(a['Jz_birth']) & np.isfinite(a['Jphi_birth'])
    if not m.any(): continue
    pot = agama.Potential(f'{PDIR}/pot_{int(pk):03d}.ini')
    Rg = pot.Rcirc(L=np.abs(a['Jphi_birth'][m]))
    ok = np.isfinite(Rg) & (Rg > 0)
    idx = np.flatnonzero(m)[ok]
    Rg_b[idx] = Rg[ok]
    zmax_b[idx] = zmax_of(pot, a['Jz_birth'][m][ok], Rg[ok])
    print(f'potential {int(pk):3d}: {ok.sum():>7,} stars  '
          f'median Rg={np.median(Rg[ok]):5.2f} kpc  median zmax={np.median(zmax_b[idx]):5.2f} kpc',
          flush=True)

pot0 = agama.Potential(f'{PDIR}/pot_127.ini')
m0 = np.isfinite(a['Jz_z0']) & np.isfinite(a['Jphi_z0'])
zmax_0 = np.full(n, np.nan, np.float32)
Rg0 = pot0.Rcirc(L=np.abs(a['Jphi_z0'][m0]))
ok0 = np.isfinite(Rg0) & (Rg0 > 0)
zmax_0[np.flatnonzero(m0)[ok0]] = zmax_of(pot0, a['Jz_z0'][m0][ok0], Rg0[ok0])

np.savez(C.OUT_DIR + '/birth_orbits_zmax.npz', ids=a['ids'], tform=a['tform'],
         zmax_birth=zmax_b, Rg_birth=Rg_b, zmax_z0=zmax_0)
f = np.isfinite(zmax_b)
print(f'\nz_max for {f.sum():,}/{n:,} ({100 * f.mean():.1f}%)')
print('z_max percentiles [kpc]:', np.round(np.percentile(zmax_b[f], [10, 25, 50, 75, 90, 99]), 2))
print(f'fraction with z_max > 2 kpc: {100 * (zmax_b[f] > 2).mean():.1f}%')
print('saved', C.OUT_DIR + '/birth_orbits_zmax.npz')
