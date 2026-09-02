"""Publication figure: birth positions on the gas, four panels.

The four-panel layout and frame of au18_birth_positions (+-25 x +-18 kpc, one
column per birth class, one row per snapshot), with the GAS surface density of
that snapshot underneath in greyscale and, on the top row, contours enclosing 50
and 90 per cent of the clean GS/E debris.

What it is for: the halo-born stars at t = 4.99 Gyr are not sprinkled through the
halo, they lie along the gas lane that bridges the host disc to the infalling
satellite, whose centroid is at (x, z) = (-10.7, -14.4) kpc.  Splitting the two
classes into their own panels keeps the gas visible, which a single overplotted
panel does not.

  disc-born   eps > 0.8  OR  z_max < 1.5 kpc
  halo-born   eps <= 0.8 AND z_max >= 1.5 kpc

Stars are drawn as scatter, not as a binned density, so that the gas underneath
stays visible everywhere rather than only where stars are absent.  N is annotated
per panel.  The gas scale is shared between the two panels of a row -- which is
the comparison the figure asks the reader to make -- but not between rows, since
the gas mass in the frame changes.

Writes Fig_paper/au18_birth_positions_gas4.pdf and .png.
"""
import gc, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
import auriga_public as ap
import config_au18 as C
import au18_frame as AF
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import orbit_tools as OT

OUT = '/data/hz420-2/EoS_explore/Fig_paper'
os.makedirs(OUT, exist_ok=True)
CUT, ZCUT = 0.8, 1.5
SNAP_A = int(sys.argv[1]) if len(sys.argv) > 1 else 72
SNAPS = [(SNAP_A, 'during the merger'), (100, 'after the merger')]
XLIM, ZLIM = 25., 18.        # widened below if the GS/E lies outside them
cG = '#8E24AA'

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 13.5, 'axes.labelsize': 15,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})


def trunc(name, lo=.22, hi=1.):
    return ListedColormap(plt.get_cmap(name)(np.linspace(lo, hi, 256)))


cD, cH = '#1F6FB2', '#FF6347'
GREY = trunc('Greys', 0., .70)

a = np.load(C.OUT_DIR + '/birth_orbits_actions.npz')
zx = np.load(C.OUT_DIR + '/birth_orbits_zmax.npz')
st = np.load(C.OUT_DIR + '/snapshot_times.npz')
SN_ALL, T_ALL = st['snaps'], st['t_snap']
GSE = np.sort(np.load(C.OUT_DIR + '/gse_clean_ids.npy'))


def frame(sn):
    sub = ap.subhalos.subfind(sn, directory=C.SIM_DIR,
                              loadlist=['SubhaloPos', 'Group_R_Crit200'])
    r200 = float(sub.data['Group_R_Crit200'][0]); cen = sub.data['SubhaloPos'][0]
    ref = ap.snapshot.load_snapshot(sn, 4, snappath=C.SIM_DIR, verbose=False,
        loadlist=['Coordinates', 'Masses', 'Potential', 'Velocities'])
    ref = ap.util.CentreOnHalo(ref, cen)
    ref = ap.util.apply_mask(ref, stars=False, radialcut=.5 * r200)
    ist, = np.where(ap.util.r(ref) < .1 * r200)
    L = np.cross(ref.data['Coordinates'][ist],
                 ref.data['Velocities'][ist] * ref.data['Masses'][ist, None])
    Ld = L.sum(0); Ld /= np.sqrt((Ld ** 2).sum())
    xd, yd, zd = ap.util.get_principal_axis(ref, ist, L=Ld)
    del ref; gc.collect()
    return cen, xd, yd, zd


def in_frame(sn, ptype, cen, xd, yd, zd, rot=None, extra=()):
    s = ap.snapshot.load_snapshot(sn, ptype, snappath=C.SIM_DIR, verbose=False,
        loadlist=['Coordinates', 'Masses', 'Velocities'] + list(extra))
    s = ap.util.CentreOnHalo(s, cen)
    ap.util.rotateto(s, xd, dir2=yd, dir3=zd)
    c = s.data['Coordinates']
    pos = np.column_stack([c[:, 1], c[:, 2], c[:, 0]]) * 1e3
    if rot is not None:
        pos = pos @ rot.T
    out = [pos, s.data['Masses'] * C.MASS_TO_MSUN] + [s.data[k] for k in extra]
    del s; gc.collect()
    return out


fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.4), sharex=True, sharey=True)
for row, (sn, when) in enumerate(SNAPS):
    k = int(np.flatnonzero(SN_ALL == sn)[0])
    t_lo, t_hi = T_ALL[k - 1], T_ALL[k]
    cen, xd, yd, zd = frame(int(sn))
    # Fix the azimuth on the GS/E so the edge-on view does not foreshorten it.
    sp0, _, sid0 = in_frame(int(sn), 4, cen, xd, yd, zd, extra=('ParticleIDs',))
    gcen, _, _, _ = AF.gse_centroid(sp0, sid0, GSE)
    # Only align when the debris still has a centroid worth pointing at.  By
    # t = 9.4 Gyr it is fully phase-mixed and its centroid sits at the origin,
    # where the azimuth it implies is pure noise.
    if np.linalg.norm(gcen) > 5.:
        ROT = AF.align_azimuth(gcen)
        print(f'snap {sn}: GS/E centroid ({gcen[0]:.1f},{gcen[1]:.1f},{gcen[2]:.1f}) '
              f'-> ({(ROT@gcen)[0]:.1f},{(ROT@gcen)[1]:.1f},{(ROT@gcen)[2]:.1f})')
        g2 = ROT @ gcen
        XLIM = max(XLIM, abs(g2[0]) + 6.); ZLIM = max(ZLIM, abs(g2[2]) + 5.)
    else:
        ROT = np.eye(3)
        print(f'snap {sn}: GS/E phase-mixed (r = {np.linalg.norm(gcen):.1f} kpc), '
              f'no azimuth alignment')
    del sp0, sid0; gc.collect()

    gpos, gm = in_frame(int(sn), 0, cen, xd, yd, zd, rot=ROT)
    nb = 300
    G2, xe, ze = np.histogram2d(gpos[:, 0], gpos[:, 2], bins=nb, weights=gm,
                                range=[[-XLIM, XLIM], [-ZLIM, ZLIM]])
    G2 = gaussian_filter(G2, 1.1) / ((2 * XLIM / nb) * (2 * ZLIM / nb))
    nz = G2[G2 > 0]
    gmin, gmax = np.percentile(nz, 55), np.percentile(nz, 99.9)
    G2 = np.where(G2 > gmin, G2, np.nan)
    del gpos, gm; gc.collect()

    spos, _, sids = in_frame(int(sn), 4, cen, xd, yd, zd, rot=ROT, extra=('ParticleIDs',))
    sel = (a['tform'] > t_lo) & (a['tform'] <= t_hi) & np.isfinite(a['eps_birth'])
    ids_w, eps_w, zmx_w = a['ids'][sel], a['eps_birth'][sel], zx['zmax_birth'][sel]
    o = np.argsort(sids); ss = sids[o]
    p = np.searchsorted(ss, ids_w)
    ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == ids_w)
    P = spos[o[p[ok]]]; eps, zmx = eps_w[ok], zmx_w[ok]
    disc = (eps > CUT) | (zmx < ZCUT)

    pg = np.searchsorted(ss, GSE)
    okg = (pg < len(ss)) & (ss[np.minimum(pg, len(ss) - 1)] == GSE)
    Gs = spos[o[pg[okg]]]
    del spos, sids; gc.collect()

    for col, (lab, m, colr, ms) in enumerate(
            [('disc-born', disc, cD, 2.2), ('halo-born', ~disc, cH, 3.6)]):
        ax = axes[row, col]
        gpc = ax.pcolormesh(xe, ze, G2.T, cmap=GREY, norm=LogNorm(vmin=gmin, vmax=gmax),
                            rasterized=True, zorder=0)
        # Alpha scaled to the sample size: low enough that overlapping markers
        # build up and the dense regions darken -- recovering the density contrast
        # a uniform-opacity scatter throws away -- but floored so the 1,214-star
        # panel does not vanish.  N ranges over a factor of 20 across the panels.
        al = float(np.clip(2500. / max(m.sum(), 1), .10, .75))
        ax.scatter(P[m, 0], P[m, 2], s=ms, c=colr, alpha=al, lw=0,
                   rasterized=True, zorder=2)
        print(f'    {lab:10s} N={m.sum():>7,}  alpha={al:.2f}')
        if row == 0:
            OT.density_contours(ax, Gs[:, 0], Gs[:, 2],
                                [[-XLIM, XLIM], [-ZLIM, ZLIM]], cG,
                                levels=(0.9, 0.5), bins=70, smooth=1.6, lw=2.0)
        ax.set(aspect='equal', xlim=(-XLIM, XLIM), ylim=(-ZLIM, ZLIM))
        bb = dict(fc='white', ec='none', alpha=.75, pad=1.8)
        ax.text(.03, .955, f'N = {m.sum():,}', transform=ax.transAxes, va='top',
                fontsize=13, bbox=bb)
        ax.text(.03, .875, f'median $|z|$ = {np.median(np.abs(P[m, 2])):.2f} kpc',
                transform=ax.transAxes, va='top', fontsize=12, bbox=bb)
        if row == 0:
            ax.text(.5, 1.04, lab, transform=ax.transAxes, ha='center', fontsize=16)
        if col == 1:
            ax.text(1.035, .5, f'$t = {t_hi:.2f}$ Gyr\n{when}', transform=ax.transAxes,
                    rotation=270, va='center', ha='left', fontsize=14)
        if col == 0:
            # lower right on the top row: the GS/E contour sits at x = -16..-6 kpc
            cax = ax.inset_axes([.44 if row == 0 else .07, .085, .52, .030])
            cb = fig.colorbar(gpc, cax=cax, orientation='horizontal')
            cb.ax.tick_params(labelsize=9.5, length=2.5, pad=1.5)
            cb.outline.set_linewidth(.7)
            cax.set_title(r'$\Sigma_{\rm gas}$ [M$_\odot$ kpc$^{-2}$]', fontsize=10, pad=3)
        if row == 0 and col == 1:
            ax.plot([], [], color=cG, lw=2.0, label='GS/E debris')
            ax.legend(loc='lower right', handlelength=1.4, borderpad=.3,
                      labelcolor='k').get_frame().set_alpha(0)
    print(f'snap {sn}: disc-born {disc.sum():,}, halo-born {(~disc).sum():,}, '
          f'gas scale {gmin:.2e}-{gmax:.2e}')
    del P, Gs; gc.collect()

for ax in axes[1, :]:
    ax.set_xlabel('$x$ [kpc]')
for ax in axes[:, 0]:
    ax.set_ylabel('$z$ [kpc]')
fig.tight_layout(pad=.5, w_pad=.6, h_pad=.6)
TAG = '' if SNAP_A == 72 else f'_snap{SNAP_A}'
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/au18_birth_positions_gas4{TAG}.{ext}', bbox_inches='tight')
print(f'\nsaved {OUT}/au18_birth_positions_gas4{TAG}.pdf and .png')
