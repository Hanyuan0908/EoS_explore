"""Wide view of the Au18 merger: gas surface density with the GS/E as contours.

The 80 kpc counterpart of diag_disc_ABC_montage.py, which shows the same
snapshots in a 40 kpc box.  Two changes, both to get the merger itself in frame
rather than only what it does to the disc:

  * W = 40 kpc, so the box is 80 kpc across and the progenitor's approach,
    passage and tail are inside it instead of crossing the edge;
  * the GS/E is drawn as density contours, not as a scatter of points.  At this
    scale the debris covers most of the panel and a scatter simply greys it out;
    contours enclosing 90, 60 and 30 per cent of the debris in frame show where
    it actually concentrates, and leave the gas map readable underneath.

The A/B/C birth populations of the 40 kpc version are dropped: they are disc-scale
and invisible here.  Grey is gas surface density, violet is the GS/E.

Writes figures/au18_disc_gse_wide_{faceon,edgeon}.png.
"""
import gc, glob, os, sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod, util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import orbit_tools as OT

os.makedirs(C.FIG_DIR, exist_ok=True)

SNAP_LIST = [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82]
W = 40.0          # half-width [kpc]; the box is 2W = 80 kpc across
NBIN = 320        # 0.25 kpc gas pixels
NMIN_CONTOUR = 50  # below this the contour is noise, so it is not drawn
CGSE = '#8E24AA'  # the paper's GS/E colour (Fig_code/CONVENTIONS.md)
GSE_IDS = np.load(C.OUT_DIR + '/gse_clean_ids.npy')
print(f'GS/E clean debris: {len(GSE_IDS):,} stars')


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
    sid = s.data['ParticleIDs']
    del s; gc.collect()

    g = snap_mod.load_snapshot(sn, 0, snappath=C.SIM_DIR, loadlist=['Coordinates', 'Masses'])
    gxyz = np.dot(g.data['Coordinates'] - cen, matrix.T) * 1000.
    gm = g.data['Masses'] * C.MASS_TO_MSUN
    del g; gc.collect()
    return sxyz, sid, gxyz, gm


def match(snapshot_ids, wanted):
    o = np.argsort(snapshot_ids); ss = snapshot_ids[o]; p = np.searchsorted(ss, wanted)
    ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == wanted)
    return o[p[ok]]


# comp 0 = disc rotation axis -> 'z'; comps 1,2 = disc plane -> 'x','y'.
PROJ = {'faceon': dict(i=1, j=2, xlab='x [kpc]', ylab='y [kpc]', tag='face-on (x-y)'),
        'edgeon': dict(i=1, j=0, xlab='x [kpc]', ylab='z [kpc]', tag='edge-on (x-z)')}
figs = {k: plt.subplots(3, 4, figsize=(17.5, 13.6), layout='constrained') for k in PROJ}

print(f'{"snap":>5s} {"t[Gyr]":>7s} {"GSE in box":>11s} {"r_med(GSE)":>11s}')
for k, sn in enumerate(SNAP_LIST):
    a = scale_factor(sn)
    t = float(C.a_to_age(a)); zred = 1. / a - 1.
    sxyz, sid, gxyz, gm = disc_frame(sn)

    iG = match(sid, GSE_IDS)
    inG = (np.abs(sxyz[iG]) < W).all(axis=1)
    gs = iG[inG]
    cube = (np.abs(gxyz) < W).all(axis=1)
    rmed = np.median(np.sqrt((sxyz[iG] ** 2).sum(1))) if len(iG) else np.nan
    print(f'{sn:5d} {t:7.3f} {inG.sum():11,d} {rmed:11.1f}')

    for key, cfg in PROJ.items():
        ax = figs[key][1].flat[k]
        i, j = cfg['i'], cfg['j']
        ax.hist2d(gxyz[cube, i], gxyz[cube, j], weights=gm[cube], bins=NBIN,
                  range=[[-W, W], [-W, W]], cmap='Greys', cmin=1., norm=LogNorm())
        if inG.sum() >= NMIN_CONTOUR:
            OT.density_contours(ax, sxyz[gs, i], sxyz[gs, j], [[-W, W], [-W, W]],
                                CGSE, levels=(0.9, 0.6, 0.3), bins=90, smooth=1.6,
                                lw=1.5, label=f'GS/E ({inG.sum():,})')
        else:
            # Not a failure: before the plunge the progenitor is simply outside an
            # 80 kpc box, and saying so is more use than an empty panel.
            ax.plot([], [], color=CGSE, lw=1.5,
                    label=f'GS/E ({inG.sum():,} in box)')
        ax.plot(0, 0, '+', color='k', ms=9, mew=1.4)
        ax.set(xlim=(-W, W), ylim=(-W, W), aspect='equal',
               title=f'snap {sn}: t={t:.2f} Gyr, z={zred:.2f}')
        ax.legend(fontsize=8, loc='upper right', framealpha=.9)
        if k // 4 == 2: ax.set_xlabel(cfg['xlab'])
        if k % 4 == 0: ax.set_ylabel(cfg['ylab'])

    del sxyz, sid, gxyz, gm; gc.collect()

for key, cfg in PROJ.items():
    fig, _ = figs[key]
    fig.suptitle(f'Au18 through the GS/E merger in an 80 kpc box, {cfg["tag"]}: gas surface '
                 f'density (grey) with the GS/E debris as contours enclosing 90, 60 and 30 '
                 f'per cent (violet)', fontsize=13)
    out = C.FIG_DIR + f'/au18_disc_gse_wide_{key}.png'
    fig.savefig(out, dpi=130)
    print('saved', out)
