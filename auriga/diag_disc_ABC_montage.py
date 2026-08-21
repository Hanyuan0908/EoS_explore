"""Where the three Eos populations sit on the gas disc through the GS/E merger.

  A (red)    born in the disc during the merger  - shown at birth
  B (purple) born hot off-plane during the merger - shown at birth
  C (orange) pre-merger disc, Splash candidate    - all members already formed are
             shown at their position in each snapshot, so the splashing is visible
  GS/E (blue) clean debris, current positions

Snapshots start at 70 so the pre-merger thin disc (C only; A and B do not exist
before snapshot 73 by construction) is visible before the plunge.
"""
import gc, glob, os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod, util

os.makedirs(C.FIG_DIR, exist_ok=True)

SNAP_LIST = [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82]
W = 20.0
NBIN = 260
GSE_IDS = np.load(C.OUT_DIR + '/gse_clean_ids.npy')
clean = np.load(C.OUT_DIR + '/eos_channels_clean.npz')
A_IDS, B_IDS = clean['A_ids'], clean['B_ids']
C_IDS = np.load(C.OUT_DIR + '/three_channels.npz')['C_ids']
print(f'A={len(A_IDS):,}  B={len(B_IDS):,}  C={len(C_IDS):,}  GS/E={len(GSE_IDS):,}')


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
figs = {k: plt.subplots(3, 4, figsize=(17.5, 13.6), layout='constrained') for k in PROJ}

print(f'{"snap":>5s} {"t[Gyr]":>7s} {"A born":>7s} {"B born":>7s} {"C here":>7s} {"GSE":>8s}')
for k, sn in enumerate(SNAP_LIST):
    a = scale_factor(sn); a_prev = scale_factor(sn - 1)
    t = float(C.a_to_age(a)); zred = 1. / a - 1.
    sxyz, sid, aform, gxyz, gm = disc_frame(sn)

    born_here = (aform > a_prev) & (aform <= a)
    iA = match(sid, A_IDS); iB = match(sid, B_IDS)
    iA = iA[born_here[iA]]; iB = iB[born_here[iB]]     # A and B shown at birth
    iC = match(sid, C_IDS)                             # C tracked once formed
    iG = match(sid, GSE_IDS)
    inC = (np.abs(sxyz[iC]) < W).all(axis=1)
    inG = (np.abs(sxyz[iG]) < W).all(axis=1)
    cube = (np.abs(gxyz) < W).all(axis=1)
    print(f'{sn:5d} {t:7.3f} {len(iA):7,d} {len(iB):7,d} {inC.sum():7,d} {inG.sum():8,d}')

    for key, cfg in PROJ.items():
        ax = figs[key][1].flat[k]
        i, j = cfg['i'], cfg['j']
        ax.hist2d(gxyz[cube, i], gxyz[cube, j], weights=gm[cube], bins=NBIN,
                  range=[[-W, W], [-W, W]], cmap='Greys', cmin=1., norm=LogNorm())
        gs = iG[inG]
        ax.scatter(sxyz[gs, i], sxyz[gs, j], s=2.6, c='#1f6fd0', alpha=.42, lw=0,
                   rasterized=True, label=f'GS/E ({inG.sum():,})')
        cs = iC[inC]
        ax.scatter(sxyz[cs, i], sxyz[cs, j], s=3.4, c='#e08214', alpha=.40, lw=0,
                   rasterized=True, label=f'C splash ({inC.sum():,})')
        ax.scatter(sxyz[iB, i], sxyz[iB, j], s=17, c='#7b3294', alpha=.9, lw=0,
                   label=f'B at birth ({len(iB):,})')
        ax.scatter(sxyz[iA, i], sxyz[iA, j], s=17, c='crimson', alpha=.9, lw=0,
                   label=f'A at birth ({len(iA):,})')
        ax.plot(0, 0, '+', color='k', ms=9, mew=1.4)
        ax.set(xlim=(-W, W), ylim=(-W, W), aspect='equal',
               title=f'snap {sn}: t={t:.2f} Gyr, z={zred:.2f}')
        if k == 0: ax.legend(fontsize=7.5, loc='upper right', markerscale=2.4, framealpha=.9)
        if k // 4 == 2: ax.set_xlabel(cfg['xlab'])
        if k % 4 == 0: ax.set_ylabel(cfg['ylab'])

    del sxyz, sid, aform, gxyz, gm; gc.collect()

for key, cfg in PROJ.items():
    fig, _ = figs[key]
    fig.suptitle(f'Au18 through the GS/E merger, {cfg["tag"]}: gas (grey), GS/E debris (blue), '
                 f'C pre-merger disc tracked (orange), A (red) and B (purple) at birth',
                 fontsize=13)
    out = C.FIG_DIR + f'/au18_disc_ABC_{key}.png'
    fig.savefig(out, dpi=130)
    print('saved', out)
