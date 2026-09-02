"""Sanity check: where were the disc-born and halo-born stars actually born?

Takes the stars whose birth circularity was measured in ONE snapshot after the
merger, splits them at eps = 0.75, and plots where they physically are in that
snapshot -- face-on and edge-on in the same disc frame the circularity was
measured in.  If the cut means anything, the eps >= 0.75 stars should lie in a
thin plane and the eps < 0.75 stars should not.

This is a check of the cut, not a result: the two populations are defined by
angular momentum, so nothing forces them to differ in position.
"""
import gc, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod

os.makedirs(C.FIG_DIR, exist_ok=True)
SNAP = int(sys.argv[1]) if len(sys.argv) > 1 else 80
CUT = 0.8
MODE = sys.argv[2] if len(sys.argv) > 2 else 'zmax'
ZCUT = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
SUF = {'eps': '_epsonly', 'epsz': '_epsz', 'zmax': ''}[MODE]
cD, cH = '#2166ac', '#b2182b'

st = np.load(C.OUT_DIR + '/snapshot_times.npz')
SNAPS, T_SNAP = st['snaps'], st['t_snap']
k = int(np.flatnonzero(SNAPS == SNAP)[0])
t_lo, t_hi = T_SNAP[k - 1], T_SNAP[k]
print(f'snapshot {SNAP}: t = {t_hi:.2f} Gyr, stars formed in {t_lo:.2f}-{t_hi:.2f} Gyr')

a = np.load(C.OUT_DIR + '/birth_orbits_actions.npz')
zx = np.load(C.OUT_DIR + '/birth_orbits_zmax.npz')
assert np.array_equal(a['ids'], zx['ids'])
sel = (a['tform'] > t_lo) & (a['tform'] <= t_hi) & np.isfinite(a['eps_birth'])
ids_w, eps_w, zmx_w = a['ids'][sel], a['eps_birth'][sel], zx['zmax_birth'][sel]
print(f'{len(ids_w):,} in-situ stars born in that interval')

s = snap_mod.load_snapshot(SNAP, 4, snappath=C.SIM_DIR,
    loadlist=['ParticleIDs', 'Coordinates', 'Velocities', 'Masses',
              'GFM_StellarFormationTime'])
real = s.data['GFM_StellarFormationTime'] > 0
sf = sub_mod.subfind(SNAP, directory=C.SIM_DIR, loadlist=['GroupFirstSub', 'SubhaloPos'])
cen = sf.data['SubhaloPos'][int(sf.data['GroupFirstSub'][0])]
x = (s.data['Coordinates'][real] - cen) * 1000.
v = s.data['Velocities'][real]; m = s.data['Masses'][real]
ids = s.data['ParticleIDs'][real]
r = np.sqrt((x * x).sum(1)); inn = r < 10.
v = v - np.average(v[inn], axis=0, weights=m[inn])
J = (m[inn, None] * np.cross(x[inn], v[inn])).sum(0); axis = J / np.linalg.norm(J)
tmp = np.array([1., 0., 0.]) if abs(axis[0]) < .9 else np.array([0., 1., 0.])
ex = np.cross(tmp, axis); ex /= np.linalg.norm(ex)
R = np.vstack([ex, np.cross(axis, ex), axis])
xr = x @ R.T
del s; gc.collect()

o = np.argsort(ids); ss = ids[o]
p = np.searchsorted(ss, ids_w)
ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == ids_w)
ix = o[p[ok]]
pos, eps, zmx = xr[ix], eps_w[ok], zmx_w[ok]
print(f'matched {ok.sum():,}')
zb = np.abs(pos[:, 2])
if MODE == 'eps':
    disc = eps > CUT
    LAB_D, LAB_H = f'$\\epsilon>{CUT}$', f'$\\epsilon\\leq{CUT}$'
else:
    zv = zb if MODE == 'epsz' else zmx
    Z = '|z|' if MODE == 'epsz' else 'z_{max}'
    disc = (eps > CUT) | (zv < ZCUT)      # rotation-supported OR vertically confined
    LAB_D = f'$\\epsilon>{CUT}$ or ${Z}<{ZCUT:g}$'
    LAB_H = f'$\\epsilon\\leq{CUT}$ and ${Z}\\geq{ZCUT:g}$'
halo = ~disc

Rc = np.hypot(pos[:, 0], pos[:, 1]); zc = zb
print(f'\n{"population":22s} {"N":>8s} {"med R":>8s} {"med |z|":>9s} {"z RMS":>8s} {"|z|<1kpc":>9s}')
for lab, mm in [('disc-born', disc), ('halo-born', halo)]:
    print(f'{lab:22s} {mm.sum():8,} {np.median(Rc[mm]):8.2f} {np.median(zc[mm]):9.2f} '
          f'{np.sqrt(np.mean(pos[mm, 2] ** 2)):8.2f} {100 * (zc[mm] < 1).mean():8.1f}%')

fig, axes = plt.subplots(2, 2, figsize=(11.4, 9.6))
FO, EO = 22., 12.
for i, (lab, mm, c) in enumerate(
        [(f'disc-born  ({LAB_D})', disc, cD), (f'halo-born  ({LAB_H})', halo, cH)]):
    for j, (proj, rng, xl, yl) in enumerate(
            [((0, 1), [[-FO, FO]] * 2, 'x [kpc]', 'y [kpc]'),
             ((0, 2), [[-FO, FO], [-EO, EO]], 'x [kpc]', 'z [kpc]')]):
        ax = axes[i, j]
        h = ax.hist2d(pos[mm, proj[0]], pos[mm, proj[1]], bins=170, range=rng,
                      cmin=1, cmap='inferno', norm=LogNorm())
        ax.set(xlabel=xl, ylabel=yl, aspect='equal',
               title=f'{lab} -- {"face-on" if j == 0 else "edge-on"}  (N={mm.sum():,})')
        if j == 1:
            ax.text(.03, .95, f'median $|z|$ = {np.median(zc[mm]):.2f} kpc\n'
                    f'RMS $z$ = {np.sqrt(np.mean(pos[mm, 2] ** 2)):.2f} kpc',
                    transform=ax.transAxes, va='top', fontsize=9, color='w')
        fig.colorbar(h[3], ax=ax, pad=.01)

fig.suptitle(f'Au18 snapshot {SNAP} (t = {t_hi:.2f} Gyr, after coalescence): '
             f'birth positions of stars formed in {t_lo:.2f}-{t_hi:.2f} Gyr', y=.985)
fig.tight_layout(rect=[0, 0, 1, .955])
out = C.FIG_DIR + f'/au18_birth_positions_snap{SNAP}{SUF}.png'
fig.savefig(out, dpi=145)
print('\nsaved', out)
