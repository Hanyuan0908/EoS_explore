"""Population C: pre-merger disc stars heated onto halo orbits by the GS/E merger.

The low-alpha Splash analogue, and the third corner of the Eos test:
  A heated disc  - born in the disc *during* the merger, heated since
  B born radial  - born hot off-plane during the merger
  C splash       - born in the disc *before* the merger, heated by it

Selection mirrors A exactly (same circularity estimator, same |z_birth| cut) but
on the pre-merger parent sample, so the only difference is when the star formed.
Birth epoch spans snapshots 62-72 (t = 3.46-4.99 Gyr); the disc spun up at
t~3.27 Gyr, so this covers essentially the whole pre-merger disc.
"""
import gc, os
import numpy as np
import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod, util

os.makedirs(C.OUT_DIR, exist_ok=True)
ids = np.load(C.OUT_DIR + '/merger_epoch_z0_samples.npz')['host_before']
print(f'pre-merger in-situ parent sample: {len(ids):,}')


def aligned_snapshot(sn, chemistry=False):
    fields = ['ParticleIDs', 'Coordinates', 'Velocities', 'Masses', 'Potential',
              'GFM_StellarFormationTime']
    if chemistry: fields += ['GFM_Metals']
    s = snap_mod.load_snapshot(sn, 4, snappath=C.SIM_DIR, loadlist=fields)
    real = s.data['GFM_StellarFormationTime'] > 0
    for key in list(s.data): s.data[key] = s.data[key][real]
    sf = sub_mod.subfind(sn, directory=C.SIM_DIR, loadlist=['GroupFirstSub', 'SubhaloPos'])
    cen = sf.data['SubhaloPos'][int(sf.data['GroupFirstSub'][0])]
    util.CentreOnHalo(s, cen)
    r0 = np.sqrt((s.data['Coordinates'] ** 2).sum(1)); inner = r0 < .01
    bulk = np.average(s.data['Velocities'][inner], axis=0, weights=s.data['Masses'][inner])
    s.data['Velocities'] -= bulk; util.align_galaxy(s, radialcut=.01)
    return s


def circularity(s):
    """Identical to ana_eos_two_channels so A, B and C share one epsilon scale."""
    x = s.data['Coordinates'] * 1000.; v = s.data['Velocities']
    r = np.sqrt((x * x).sum(1))
    R = np.hypot(x[:, 1], x[:, 2]); jz = x[:, 1] * v[:, 2] - x[:, 2] * v[:, 1]
    disc = (R > 3) & (R < 12) & (np.abs(x[:, 0]) < 2)
    if np.median(jz[disc]) < 0: jz *= -1
    E = .5 * (v * v).sum(1) + s.data['Potential']
    valid = np.isfinite(E) & np.isfinite(jz) & (r < 50)
    edges = np.quantile(E[valid], np.linspace(0, 1, 241))
    ib = np.clip(np.searchsorted(edges, E, 'right') - 1, 0, 239)
    jc = np.full(240, np.nan)
    for b in range(240):
        q = valid & (ib == b) & (jz > 0)
        if q.sum() > 30: jc[b] = np.percentile(jz[q], 95)
    ok = np.isfinite(jc); jc = np.interp(np.arange(240), np.flatnonzero(ok), jc[ok])
    return jz / jc[ib], r, np.abs(x[:, 0]), np.hypot(x[:, 1], x[:, 2])


def match(snapshot_ids, wanted):
    o = np.argsort(snapshot_ids); ss = snapshot_ids[o]; p = np.searchsorted(ss, wanted)
    ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == wanted)
    return o[p[ok]], ok


# z=0: present-day circularity, radius, chemistry, formation time.
s0 = aligned_snapshot(127, chemistry=True)
ix, ok = match(s0.data['ParticleIDs'], ids)
ids = ids[ok]
eps0, r0, _, _ = circularity(s0)
eps_z0 = eps0[ix]; r_z0 = r0[ix]
aform = s0.data['GFM_StellarFormationTime'][ix]
metals = s0.data['GFM_Metals'][ix]
del s0, eps0, r0; gc.collect()
tform = C.a_to_age(aform)
print(f'matched at z=0: {len(ids):,}; t_birth {tform.min():.2f}-{tform.max():.2f} Gyr')

snaps = np.arange(62, 73)
snap_a = []
for sn in snaps:
    tmp = snap_mod.load_snapshot(int(sn), 4, snappath=C.SIM_DIR,
                                 loadlist=['GFM_StellarFormationTime'])
    snap_a.append(float(tmp.time)); del tmp
assigned = np.clip(np.searchsorted(np.asarray(snap_a), aform), 0, len(snaps) - 1)

eps_birth = np.full(len(ids), np.nan)
z_birth = np.full(len(ids), np.nan)
R_birth = np.full(len(ids), np.nan)
for k, sn in enumerate(snaps):
    rows = np.flatnonzero(assigned == k)
    if not len(rows): continue
    s = aligned_snapshot(int(sn))
    ep, _, zz, RR = circularity(s)
    ix, ok = match(s.data['ParticleIDs'], ids[rows])
    eps_birth[rows[ok]] = ep[ix]; z_birth[rows[ok]] = zz[ix]; R_birth[rows[ok]] = RR[ix]
    print(f'snap {sn}: recovered {ok.sum():,}/{len(rows):,}', flush=True)
    del s, ep, zz, RR; gc.collect()

feh = C.bracket_abundance(metals, 'Fe', 'H')
ELS = ['C', 'N', 'O', 'Ne', 'Mg', 'Si']
ratios = {e: C.bracket_abundance(metals, e, 'Fe') for e in ELS}

out = C.OUT_DIR + '/premerger_splash.npz'
np.savez(out, ids=ids, eps_birth=eps_birth, eps_z0=eps_z0, r_z0=r_z0,
         z_birth=z_birth, R_birth=R_birth, tform=tform, feh=feh,
         **{e.lower() + 'fe': ratios[e] for e in ELS})

good = np.isfinite(eps_birth) & np.isfinite(eps_z0)
Csel = good & (eps_birth > .7) & (z_birth < 1.) & (eps_z0 < .3)
print(f'\nwith birth kinematics: {good.sum():,}')
print(f'C splash (eps_b>0.7, |z_b|<1, eps_0<0.3): {Csel.sum():,}')
print(f'  median t_birth={np.median(tform[Csel]):.2f} Gyr, R_birth={np.median(R_birth[Csel]):.2f} kpc, '
      f'r_z0={np.median(r_z0[Csel]):.2f} kpc, [Fe/H]={np.median(feh[Csel]):+.3f}')
print('saved', out)
