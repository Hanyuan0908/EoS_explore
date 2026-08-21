"""Where the cleaned Eos channels are born, on the gas disc.

Same frames as diag_disc_gas_gse_montage, but instead of every newborn star only
the cleaned channel members are shown, at their birth snapshot:
  A (red)    = eps_birth>0.7 -> eps_z0<0.3, |z_birth| < 1 kpc  (heated disc)
  B (purple) = eps_birth<0.3 -> eps_z0<0.3, |z_birth| > 3 kpc  (born radial)
Snapshots start at 73 because the parent sample is defined from births after
snapshot 72; earlier panels would be empty by construction.
"""
import gc, glob, os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod, util

os.makedirs(C.FIG_DIR, exist_ok=True)

SNAP_LIST = [73, 74, 75, 76, 77, 78, 79, 80, 81, 82]
W = 20.0
NBIN = 260
GSE_IDS = np.load(C.OUT_DIR + '/gse_clean_ids.npy')

clean = np.load(C.OUT_DIR + '/eos_channels_clean.npz')
A_IDS, B_IDS = clean['A_ids'], clean['B_ids']
print(f'cleaned samples: A={len(A_IDS):,}  B={len(B_IDS):,}')


def scale_factor(sn):
    f = sorted(glob.glob(f'{C.SIM_DIR}/snapdir_{sn:03d}/snapshot_{sn:03d}.*.hdf5'))[0]
    with h5py.File(f, 'r') as h:
        return float(h['Header'].attrs['Time'])


def disc_frame(sn):
    """Halo-centred, disc-aligned star and gas coordinates in kpc (align_galaxy convention)."""
    s = snap_mod.load_snapshot(sn, 4, snappath=C.SIM_DIR,
        loadlist=['ParticleIDs', 'Coordinates', 'Velocities', 'Masses',
                  'GFM_StellarFormationTime'])
    real = s.data['GFM_StellarFormationTime'] > 0
    for k in list(s.data): s.data[k] = s.data[k][real]
    sf = sub_mod.subfind(sn, directory=C.SIM_DIR, loadlist=['GroupFirstSub', 'SubhaloPos'])
    cen = sf.data['SubhaloPos'][int(sf.data['GroupFirstSub'][0])]
    util.CentreOnHalo(s, cen)
    rr = np.sqrt((s.data['Coordinates'] ** 2).sum(1))
    idx, = np.where(rr < .01)
    bulk = np.average(s.data['Velocities'][idx], axis=0, weights=s.data['Masses'][idx])
    s.data['Velocities'] -= bulk
    L = np.cross(s.data['Coordinates'][idx, :],
                 s.data['Velocities'][idx, :] * s.data['Masses'][idx, None]).sum(axis=0)
    xdir, ydir, zdir = util.get_principal_axis(s, idx, L=L / np.sqrt((L ** 2).sum()))
    matrix = np.array([xdir, ydir, zdir])

    sxyz = np.dot(s.data['Coordinates'], matrix.T) * 1000.
    sid = s.data['ParticleIDs']; aform = s.data['GFM_StellarFormationTime']
    del s; gc.collect()

    g = snap_mod.load_snapshot(sn, 0, snappath=C.SIM_DIR, loadlist=['Coordinates', 'Masses'])
    gxyz = np.dot(g.data['Coordinates'] - cen, matrix.T) * 1000.
    gm = g.data['Masses'] * C.MASS_TO_MSUN
    del g; gc.collect()
    return sxyz, sid, aform, gxyz, gm


def match(snapshot_ids, wanted):
    o = np.argsort(snapshot_ids); ss = snapshot_ids[o]; p = np.searchsorted(ss, wanted)
    ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == wanted)
    return o[p[ok]]


PROJ = {'faceon': dict(i=1, j=2, xlab='x [kpc]', ylab='y [kpc]', tag='face-on (x-y)'),
        'edgeon': dict(i=1, j=0, xlab='x [kpc]', ylab='z [kpc]', tag='edge-on (x-z)')}
figs = {k: plt.subplots(2, 5, figsize=(21, 10.2), layout='constrained') for k in PROJ}
payload = {}   # cached so the figures can be restyled without re-reading snapshots

print(f'{"snap":>5s} {"t[Gyr]":>7s} {"A born":>7s} {"B born":>7s} {"GSE<20":>8s}')
for k, sn in enumerate(SNAP_LIST):
    a = scale_factor(sn); a_prev = scale_factor(sn - 1)
    t = float(C.a_to_age(a)); zred = 1. / a - 1.
    sxyz, sid, aform, gxyz, gm = disc_frame(sn)

    born_here = (aform > a_prev) & (aform <= a)
    iA = match(sid, A_IDS); iB = match(sid, B_IDS); iG = match(sid, GSE_IDS)
    iA = iA[born_here[iA]]; iB = iB[born_here[iB]]      # show each star at its birth
    inG = (np.abs(sxyz[iG]) < W).all(axis=1)
    cube = (np.abs(gxyz) < W).all(axis=1)
    print(f'{sn:5d} {t:7.3f} {len(iA):7,d} {len(iB):7,d} {inG.sum():8,d}')

    for key, cfg in PROJ.items():
        ax = figs[key][1].flat[k]
        i, j = cfg['i'], cfg['j']
        ax.hist2d(gxyz[cube, i], gxyz[cube, j], weights=gm[cube], bins=NBIN,
                  range=[[-W, W], [-W, W]], cmap='Greys', cmin=1., norm=LogNorm())
        gs = iG[inG]
        ax.scatter(sxyz[gs, i], sxyz[gs, j], s=1.2, c='#7fb3e0', alpha=.30, lw=0,
                   rasterized=True, label=f'GS/E debris ({inG.sum():,})')
        ax.scatter(sxyz[iB, i], sxyz[iB, j], s=17, c='#7b3294', alpha=.9, lw=0,
                   label=f'B born radial ({len(iB):,})')
        ax.scatter(sxyz[iA, i], sxyz[iA, j], s=17, c='crimson', alpha=.9, lw=0,
                   label=f'A heated disc ({len(iA):,})')
        ax.plot(0, 0, '+', color='k', ms=9, mew=1.4)
        ax.set(xlim=(-W, W), ylim=(-W, W), aspect='equal',
               title=f'snap {sn}: t={t:.2f} Gyr, z={zred:.2f}')
        if k == 0: ax.legend(fontsize=7.5, loc='upper right', markerscale=2.2, framealpha=.88)
        if k // 5 == 1: ax.set_xlabel(cfg['xlab'])
        if k % 5 == 0: ax.set_ylabel(cfg['ylab'])

    payload[f'{sn}_gas'] = gxyz[cube]; payload[f'{sn}_gasm'] = gm[cube]
    payload[f'{sn}_A'] = sxyz[iA]; payload[f'{sn}_B'] = sxyz[iB]
    payload[f'{sn}_G'] = sxyz[iG[inG]]; payload[f'{sn}_t'] = np.array(t)
    del sxyz, sid, aform, gxyz, gm; gc.collect()

np.savez_compressed(C.OUT_DIR + '/disc_AB_montage_payload.npz', snaps=np.array(SNAP_LIST), **payload)

for key, cfg in PROJ.items():
    fig, _ = figs[key]
    fig.suptitle(f'Au18 cleaned Eos channels at birth, {cfg["tag"]}: gas surface density (grey), '
                 f'GS/E debris (pale blue), A heated disc |z_b|<1 kpc (red), '
                 f'B born radial |z_b|>3 kpc (purple)', fontsize=13)
    out = C.FIG_DIR + f'/au18_disc_AB_{key}.png'
    fig.savefig(out, dpi=130)
    print('saved', out)
