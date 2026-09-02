"""Is the central 'halo-born' population just the stellar bar?

Measures the m=2 Fourier mode of the existing stellar disc at one snapshot --
A2/A0 gives the bar strength, and its phase gives the bar position angle -- then
measures the position angle of the newborn stars classified halo-born (eps < 0.75)
in the same frame.  Bar orbits are elongated and planar with low angular momentum
for their energy, so they land at low eps even though they are disc structure, not
halo structure; if the two angles agree the cut is picking up the bar.

Both angles are measured in the same rotated frame, so the arbitrary azimuthal
zero point cancels in the difference.
"""
import gc, os, sys
import numpy as np
import matplotlib.pyplot as plt
import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod

os.makedirs(C.FIG_DIR, exist_ok=True)
SNAP = int(sys.argv[1]) if len(sys.argv) > 1 else 80
CUT, ZCUT = 0.8, 2.0

st = np.load(C.OUT_DIR + '/snapshot_times.npz')
SNAPS, T_SNAP = st['snaps'], st['t_snap']
k = int(np.flatnonzero(SNAPS == SNAP)[0])
t_lo, t_hi = T_SNAP[k - 1], T_SNAP[k]

s = snap_mod.load_snapshot(SNAP, 4, snappath=C.SIM_DIR,
    loadlist=['ParticleIDs', 'Coordinates', 'Velocities', 'Masses',
              'GFM_StellarFormationTime'])
real = s.data['GFM_StellarFormationTime'] > 0
sf = sub_mod.subfind(SNAP, directory=C.SIM_DIR, loadlist=['GroupFirstSub', 'SubhaloPos'])
cen = sf.data['SubhaloPos'][int(sf.data['GroupFirstSub'][0])]
x = (s.data['Coordinates'][real] - cen) * 1000.
v = s.data['Velocities'][real]; m = s.data['Masses'][real]
ids = s.data['ParticleIDs'][real]
tform_all = C.a_to_age(s.data['GFM_StellarFormationTime'][real])
r = np.sqrt((x * x).sum(1)); inn = r < 10.
v = v - np.average(v[inn], axis=0, weights=m[inn])
J = (m[inn, None] * np.cross(x[inn], v[inn])).sum(0); axis = J / np.linalg.norm(J)
tmp = np.array([1., 0., 0.]) if abs(axis[0]) < .9 else np.array([0., 1., 0.])
ex = np.cross(tmp, axis); ex /= np.linalg.norm(ex)
Rot = np.vstack([ex, np.cross(axis, ex), axis])
xr = x @ Rot.T
del s; gc.collect()

R = np.hypot(xr[:, 0], xr[:, 1]); phi = np.arctan2(xr[:, 1], xr[:, 0]); z = xr[:, 2]


def m2(Rv, phiv, wv):
    """A2/A0 and the m=2 position angle, in degrees, for one set of particles."""
    if len(Rv) < 50: return np.nan, np.nan
    c = np.sum(wv * np.exp(2j * phiv))
    return np.abs(c) / np.sum(wv), np.degrees(.5 * np.angle(c)) % 180


# --- the bar, from the pre-existing stellar disc (older than the newborns) -----
old = (np.abs(z) < 3) & (tform_all < t_lo)
print(f'existing stellar disc at snap {SNAP}: {old.sum():,} stars with |z| < 3 kpc\n')
print(f'{"R [kpc]":>10s} {"A2/A0":>8s} {"PA [deg]":>9s} {"N":>9s}')
edges = np.arange(0, 10.1, 1.)
bar_pa, bar_amp = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    q = old & (R >= lo) & (R < hi)
    A, pa = m2(R[q], phi[q], m[q])
    print(f'{lo:4.0f}-{hi:<5.0f} {A:8.3f} {pa:9.1f} {q.sum():9,}')
    if 1 <= lo < 5 and np.isfinite(A):
        bar_pa.append(pa); bar_amp.append(A)
BAR_PA = np.degrees(.5 * np.angle(np.mean(np.exp(2j * np.radians(bar_pa)))))% 180
print(f'\nbar: mean A2/A0 over 1-5 kpc = {np.mean(bar_amp):.3f}, position angle = {BAR_PA:.1f} deg')

# --- the newborn stars, split by the circularity cut --------------------------
a = np.load(C.OUT_DIR + '/birth_orbits_agama.npz')
sel = (a['tform'] > t_lo) & (a['tform'] <= t_hi) & np.isfinite(a['eps_birth'])
o = np.argsort(ids); ss = ids[o]
p = np.searchsorted(ss, a['ids'][sel])
ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == a['ids'][sel])
ix = o[p[ok]]; eps = a['eps_birth'][sel][ok]
zb = np.abs(z[ix])
is_disc = (eps > CUT) | (zb < ZCUT)       # rotation-supported OR planar
is_halo = ~is_disc
print(f'\nnewborn stars ({t_lo:.2f}-{t_hi:.2f} Gyr): {len(ix):,}')
print(f'{"population":26s} {"R range":>9s} {"A2/A0":>8s} {"PA [deg]":>9s} {"PA - bar PA":>12s} {"N":>8s}')
for lab, mm in [('halo-born', is_halo), ('disc-born', is_disc)]:
    for lo, hi in [(0, 3), (3, 10)]:
        q = mm & (R[ix] >= lo) & (R[ix] < hi) & (np.abs(z[ix]) < 3)
        A, pa = m2(R[ix][q], phi[ix][q], m[ix][q])
        d = (pa - BAR_PA + 90) % 180 - 90
        print(f'{lab:26s} {f"{lo}-{hi}":>9s} {A:8.3f} {pa:9.1f} {d:+11.1f} {q.sum():8,}')

# inertia tensor of the inner halo-born stars, an independent angle estimate
q = is_halo & (R[ix] < 3) & (np.abs(z[ix]) < 3)
P = xr[ix][q][:, :2]
w, V = np.linalg.eigh(np.cov(P.T))
pa_in = np.degrees(np.arctan2(V[1, -1], V[0, -1])) % 180
if q.sum() < 20:
    print('\nfewer than 20 inner halo-born stars: the bar no longer contaminates them')
    raise SystemExit
print(f'\ninner halo-born stars, inertia-tensor major axis = {pa_in:.1f} deg '
      f'(bar {BAR_PA:.1f}, difference {((pa_in - BAR_PA + 90) % 180 - 90):+.1f} deg); '
      f'axis ratio b/a = {np.sqrt(w[0] / w[1]):.2f}')
