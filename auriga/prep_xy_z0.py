"""In-plane positions at z=0, so a solar-neighbourhood selection can be applied.

The z=0 catalogue stores R and z but not the azimuth, which is all that is needed
until one wants to place an observer somewhere in the disc.  This saves the two
in-plane coordinates for every in-situ star and for the GS/E debris, in the same
disc-aligned frame used everywhere else (align_galaxy puts the symmetry axis on
component 0, so the disc plane is components 1 and 2).

Writes out/z0_xy.npz.
"""
import gc, os
import numpy as np
import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod, util

s = snap_mod.load_snapshot(127, 4, snappath=C.SIM_DIR,
    loadlist=['ParticleIDs', 'Coordinates', 'Velocities', 'Masses',
              'GFM_StellarFormationTime'])
real = s.data['GFM_StellarFormationTime'] > 0
for k in list(s.data):
    s.data[k] = s.data[k][real]
sf = sub_mod.subfind(127, directory=C.SIM_DIR, loadlist=['GroupFirstSub', 'SubhaloPos'])
cen = sf.data['SubhaloPos'][int(sf.data['GroupFirstSub'][0])]
util.CentreOnHalo(s, cen)
rr = np.sqrt((s.data['Coordinates'] ** 2).sum(1))
idx = np.flatnonzero(rr < .01)
s.data['Velocities'] -= np.average(s.data['Velocities'][idx], axis=0,
                                   weights=s.data['Masses'][idx])
util.align_galaxy(s, radialcut=.01)

xyz = s.data['Coordinates'] * 1000.
sid = s.data['ParticleIDs']
# disc plane = components (1, 2); component 0 is the symmetry axis
np.savez(C.OUT_DIR + '/z0_xy.npz', ids=sid,
         x=xyz[:, 1].astype(np.float32), y=xyz[:, 2].astype(np.float32),
         z=xyz[:, 0].astype(np.float32))
print(f'saved in-plane positions for {len(sid):,} stars')
print(f'  x range {xyz[:, 1].min():.1f} to {xyz[:, 1].max():.1f} kpc')
print('saved', C.OUT_DIR + '/z0_xy.npz')
