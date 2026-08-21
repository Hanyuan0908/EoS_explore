"""z=0 reference populations for the chemistry comparison: GS/E debris and the cold disc.

GS/E is the clean debris ID list; the disc is every in-situ star that is still
rotationally supported at z=0 (eps_z0 > 0.7).  Both provide the chemical backdrop
against which the A/B/C channels are compared.
"""
import gc, os
import numpy as np
import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod, util

os.makedirs(C.OUT_DIR, exist_ok=True)
ELS = ['C', 'N', 'O', 'Ne', 'Mg', 'Si']

s = snap_mod.load_snapshot(127, 4, snappath=C.SIM_DIR,
    loadlist=['ParticleIDs', 'Coordinates', 'Velocities', 'Masses', 'Potential',
              'GFM_StellarFormationTime', 'GFM_Metals'])
real = s.data['GFM_StellarFormationTime'] > 0
# matched_z0 indices refer to the untrimmed snapshot arrays.
old_to_real = np.full(len(real), -1, np.int64)
old_to_real[np.flatnonzero(real)] = np.arange(int(real.sum()))
insitu_idx = old_to_real[np.load(C.OUT_DIR + '/matched_z0.npz')['ii']]
insitu_idx = insitu_idx[insitu_idx >= 0]

for k in list(s.data): s.data[k] = s.data[k][real]
sf = sub_mod.subfind(127, directory=C.SIM_DIR, loadlist=['GroupFirstSub', 'SubhaloPos'])
cen = sf.data['SubhaloPos'][int(sf.data['GroupFirstSub'][0])]
util.CentreOnHalo(s, cen)
rr = np.sqrt((s.data['Coordinates'] ** 2).sum(1)); inner = rr < .01
bulk = np.average(s.data['Velocities'][inner], axis=0, weights=s.data['Masses'][inner])
s.data['Velocities'] -= bulk
util.align_galaxy(s, radialcut=.01)

x = s.data['Coordinates'] * 1000.; v = s.data['Velocities']
r = np.sqrt((x * x).sum(1))
R = np.hypot(x[:, 1], x[:, 2]); jz = x[:, 1] * v[:, 2] - x[:, 2] * v[:, 1]
disc_ref = (R > 3) & (R < 12) & (np.abs(x[:, 0]) < 2)
if np.median(jz[disc_ref]) < 0: jz *= -1
E = .5 * (v * v).sum(1) + s.data['Potential']
valid = np.isfinite(E) & np.isfinite(jz) & (r < 50)
edges = np.quantile(E[valid], np.linspace(0, 1, 241))
ib = np.clip(np.searchsorted(edges, E, 'right') - 1, 0, 239)
jc = np.full(240, np.nan)
for b in range(240):
    q = valid & (ib == b) & (jz > 0)
    if q.sum() > 30: jc[b] = np.percentile(jz[q], 95)
ok = np.isfinite(jc); jc = np.interp(np.arange(240), np.flatnonzero(ok), jc[ok])
eps = jz / jc[ib]

sid = s.data['ParticleIDs']
gse_ids = np.load(C.OUT_DIR + '/gse_clean_ids.npy')
o = np.argsort(sid); ss = sid[o]; p = np.searchsorted(ss, gse_ids)
m = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == gse_ids)
gse_idx = o[p[m]]

insitu = np.zeros(len(sid), bool); insitu[insitu_idx] = True
disc_idx = np.flatnonzero(insitu & (eps > .7) & np.isfinite(eps) & (r < 30))
print(f'GS/E at z=0: {len(gse_idx):,};  cold in-situ disc (eps>0.7, r<30): {len(disc_idx):,}')

metals = s.data['GFM_Metals']
out = {}
for tag, idx in [('gse', gse_idx), ('disc', disc_idx)]:
    out[f'{tag}_feh'] = C.bracket_abundance(metals[idx], 'Fe', 'H')
    for e in ELS:
        out[f'{tag}_{e.lower()}fe'] = C.bracket_abundance(metals[idx], e, 'Fe')
    out[f'{tag}_eps'] = eps[idx]; out[f'{tag}_r'] = r[idx]
    # A handful of very metal-poor GS/E particles have zero Fe -> -inf.
    print(f'  {tag}: median [Fe/H]={np.nanmedian(out[f"{tag}_feh"]):+.3f}, '
          f'[Mg/Fe]={np.nanmedian(out[f"{tag}_mgfe"]):+.3f}')

np.savez(C.OUT_DIR + '/z0_reference_pops.npz', **out)
print('saved', C.OUT_DIR + '/z0_reference_pops.npz')
