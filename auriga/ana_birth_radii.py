"""Birth radii for the merger-born in-situ sample (companion to the birth kinematics).

Chemistry is frozen at birth, so any [Fe/H] difference between the Eos channels has
to be tested against the birth-epoch radial gradient, not the z=0 radius.
"""
import gc, os
import numpy as np
import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod, util

os.makedirs(C.OUT_DIR, exist_ok=True)
ids = np.load(C.OUT_DIR + '/eos_two_channels.npz')['ids']


def aligned_cyl(sn):
    s = snap_mod.load_snapshot(sn, 4, snappath=C.SIM_DIR,
        loadlist=['ParticleIDs', 'Coordinates', 'Velocities', 'Masses', 'GFM_StellarFormationTime'])
    real = s.data['GFM_StellarFormationTime'] > 0
    for k in list(s.data): s.data[k] = s.data[k][real]
    sf = sub_mod.subfind(sn, directory=C.SIM_DIR, loadlist=['GroupFirstSub', 'SubhaloPos'])
    cen = sf.data['SubhaloPos'][int(sf.data['GroupFirstSub'][0])]
    util.CentreOnHalo(s, cen)
    rr = np.sqrt((s.data['Coordinates'] ** 2).sum(1)); q = rr < .01
    bulk = np.average(s.data['Velocities'][q], axis=0, weights=s.data['Masses'][q])
    s.data['Velocities'] -= bulk; util.align_galaxy(s, radialcut=.01)
    x = s.data['Coordinates'] * 1000.
    # align_galaxy puts the disc angular momentum on component 0 -> disc plane = (1,2).
    return s.data['ParticleIDs'], np.hypot(x[:, 1], x[:, 2]), np.abs(x[:, 0]), np.sqrt((x * x).sum(1))


def match(snapshot_ids, wanted):
    o = np.argsort(snapshot_ids); ss = snapshot_ids[o]; p = np.searchsorted(ss, wanted)
    ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == wanted)
    return o[p[ok]], ok


# Same snapshot assignment as ana_merger_birth_vs_z0_kinematics / ana_eos_two_channels.
snapshots = np.arange(73, 83)
ascale = []
for sn in snapshots:
    tmp = snap_mod.load_snapshot(int(sn), 4, snappath=C.SIM_DIR, loadlist=['GFM_StellarFormationTime'])
    ascale.append(float(tmp.time)); del tmp
ascale = np.asarray(ascale)

s0 = snap_mod.load_snapshot(127, 4, snappath=C.SIM_DIR, loadlist=['ParticleIDs', 'GFM_StellarFormationTime'])
real = s0.data['GFM_StellarFormationTime'] > 0
sid0 = s0.data['ParticleIDs'][real]; a0 = s0.data['GFM_StellarFormationTime'][real]
ix, ok = match(sid0, ids)
ids = ids[ok]; aform = a0[ix]; del s0, sid0, a0; gc.collect()
tform = C.a_to_age(aform)
assigned = np.clip(np.searchsorted(ascale, aform), 0, len(snapshots) - 1)

R_birth = np.full(len(ids), np.nan)
z_birth = np.full(len(ids), np.nan)
r_birth = np.full(len(ids), np.nan)
snap_birth = np.zeros(len(ids), int)
for k, sn in enumerate(snapshots):
    rows = np.flatnonzero(assigned == k)
    if not len(rows): continue
    sid, R, zz, rr = aligned_cyl(int(sn))
    ix, ok = match(sid, ids[rows])
    R_birth[rows[ok]] = R[ix]; z_birth[rows[ok]] = zz[ix]; r_birth[rows[ok]] = rr[ix]
    snap_birth[rows] = sn
    print(f'snap {sn}: birth radii recovered {ok.sum():,}/{len(rows):,}', flush=True)
    del sid, R, zz, rr; gc.collect()

out = C.OUT_DIR + '/merger_birth_radii.npz'
np.savez(out, ids=ids, R_birth=R_birth, z_birth=z_birth, r_birth=r_birth,
         tform=tform, aform=aform, snap_birth=snap_birth)
good = np.isfinite(R_birth)
print(f'saved {out}: {good.sum():,}/{len(ids):,} with birth radii; '
      f'median R_birth={np.nanmedian(R_birth):.2f} kpc, tform range '
      f'{tform.min():.2f}-{tform.max():.2f} Gyr')
