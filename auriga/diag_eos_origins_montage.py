"""Where the two Eos populations are born, through the GS/E merger.

Same style as diag_disc_ABC_montage.py, but for the split that the birth
v_R-v_phi plane reveals (see eos_origins.py): the stars that are Eos-like at z=0
divide into one lobe born already hot and one born on the disc ridge.

  all newborn (black)  every in-situ star formed between this snapshot and the
                       last, so the two Eos populations can be seen against the
                       star formation they are drawn from
  halo-born (purple)   Eos-like at z=0, v_phi,birth < 150 km/s - shown at birth
  disc-born (crimson)  Eos-like at z=0, v_phi,birth >= 150 km/s - shown at birth
  GS/E (blue)          clean debris, current positions
  gas (grey)           surface density background
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
org = np.load(C.OUT_DIR + '/eos_two_origins.npz')
HALO_IDS, DISC_IDS = org['halo_born_ids'], org['disc_born_ids']
print(f'halo-born={len(HALO_IDS):,}  disc-born={len(DISC_IDS):,}  GS/E={len(GSE_IDS):,}')


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

print(f'{"snap":>5s} {"t[Gyr]":>7s} {"newborn":>8s} {"halo":>6s} {"disc":>6s} {"GSE":>8s}')
for k, sn in enumerate(SNAP_LIST):
    a = scale_factor(sn); a_prev = scale_factor(sn - 1)
    t = float(C.a_to_age(a)); zred = 1. / a - 1.
    sxyz, sid, aform, gxyz, gm = disc_frame(sn)

    born_here = (aform > a_prev) & (aform <= a)
    iN = np.flatnonzero(born_here & (np.abs(sxyz) < W).all(axis=1))   # all newborn stars
    iH = match(sid, HALO_IDS); iD = match(sid, DISC_IDS)
    iH = iH[born_here[iH]]; iD = iD[born_here[iD]]     # both shown at birth
    iG = match(sid, GSE_IDS)
    inG = (np.abs(sxyz[iG]) < W).all(axis=1)
    cube = (np.abs(gxyz) < W).all(axis=1)
    print(f'{sn:5d} {t:7.3f} {len(iN):8,d} {len(iH):6,d} {len(iD):6,d} {inG.sum():8,d}')

    for key, cfg in PROJ.items():
        ax = figs[key][1].flat[k]
        i, j = cfg['i'], cfg['j']
        ax.hist2d(gxyz[cube, i], gxyz[cube, j], weights=gm[cube], bins=NBIN,
                  range=[[-W, W], [-W, W]], cmap='Greys', cmin=1., norm=LogNorm())
        ax.scatter(sxyz[iN, i], sxyz[iN, j], s=1.6, c='k', alpha=.20, lw=0,
                   rasterized=True, label=f'all newborn ({len(iN):,})')
        gs = iG[inG]
        ax.scatter(sxyz[gs, i], sxyz[gs, j], s=2.6, c='#1f6fd0', alpha=.42, lw=0,
                   rasterized=True, label=f'GS/E ({inG.sum():,})')
        ax.scatter(sxyz[iH, i], sxyz[iH, j], s=19, c='#7b3294', alpha=.9, lw=0,
                   label=f'halo-born Eos ({len(iH):,})')
        ax.scatter(sxyz[iD, i], sxyz[iD, j], s=19, c='crimson', alpha=.9, lw=0,
                   label=f'disc-born Eos ({len(iD):,})')
        ax.plot(0, 0, '+', color='k', ms=9, mew=1.4)
        ax.set(xlim=(-W, W), ylim=(-W, W), aspect='equal',
               title=f'snap {sn}: t={t:.2f} Gyr, z={zred:.2f}')
        if k == 0: ax.legend(fontsize=7.5, loc='upper right', markerscale=2.4, framealpha=.9)
        if k // 4 == 2: ax.set_xlabel(cfg['xlab'])
        if k % 4 == 0: ax.set_ylabel(cfg['ylab'])

    del sxyz, sid, aform, gxyz, gm; gc.collect()

for key, cfg in PROJ.items():
    fig, _ = figs[key]
    fig.suptitle(f'Au18 through the GS/E merger, {cfg["tag"]}: gas (grey), all newborn stars '
                 f'(black), GS/E debris (blue), and the two Eos populations at birth -- '
                 f'halo-born (purple), disc-born (crimson)', fontsize=13)
    out = C.FIG_DIR + f'/au18_eos_origins_{key}.png'
    fig.savefig(out, dpi=130)
    print('saved', out)
