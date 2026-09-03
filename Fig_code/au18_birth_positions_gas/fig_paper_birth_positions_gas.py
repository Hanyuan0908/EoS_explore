"""Publication figure: the halo-born stars form in the gas lane joining GS/E to the host.

A companion to au18_birth_positions, built to show the mechanism rather than just
the outcome.  Each panel is the edge-on GAS surface density of one snapshot in
greyscale, with the stars formed in that snapshot (~0.15 Gyr) scattered on top,
split by the same criterion:

  disc-born   eps > 0.8  OR  z_max < 1.5 kpc
  halo-born   eps <= 0.8 AND z_max >= 1.5 kpc

Top: t = 4.99 Gyr, the GS/E pericentre passage.  The violet contours enclose 50
and 90 per cent of the clean GS/E debris (out/gse_clean_ids.npy), which at this
moment sits at (x, z) = (-10.7, -14.4) kpc, r = 20 kpc.  The halo-born stars are
not spread over the halo at random: they lie along the gas that bridges the
satellite to the host disc.

Bottom: t = 9.41 Gyr, long after coalescence, as a control -- the gas has settled
back to a thin disc and almost no halo-born stars form.

Writes Fig_paper/au18_birth_positions_gas.pdf and .png.
"""
import gc, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import auriga_public as ap
import config_au18 as C
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import orbit_tools as OT

OUT = '/data/hz420-2/EoS_explore/Fig_paper'
os.makedirs(OUT, exist_ok=True)
CUT, ZCUT = 0.8, 1.5
SNAPS = [(72, 'GS/E pericentre'), (100, 'after the merger')]
XLIM, ZLIM = 30., 24.
cD, cH, cG = '#1F6FB2', '#FF6347', '#8E24AA'
# Greys capped below black: the densest gas should read as dark grey so the
# orange scatter on top of the disc stays visible.
GREY = ListedColormap(plt.get_cmap('Greys')(np.linspace(0., .82, 256)))

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

a = np.load(C.OUT_DIR + '/birth_orbits_actions.npz')
zx = np.load(C.OUT_DIR + '/birth_orbits_zmax.npz')
st = np.load(C.OUT_DIR + '/snapshot_times.npz')
SN_ALL, T_ALL = st['snaps'], st['t_snap']
GSE = np.sort(np.load(C.OUT_DIR + '/gse_clean_ids.npy'))


def frame(sn):
    """Centre and principal-axis rotation, from the stars, as in prep_birth_actions."""
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


def in_frame(sn, ptype, cen, xd, yd, zd, extra=()):
    s = ap.snapshot.load_snapshot(sn, ptype, snappath=C.SIM_DIR, verbose=False,
        loadlist=['Coordinates', 'Masses', 'Velocities'] + list(extra))
    s = ap.util.CentreOnHalo(s, cen)
    ap.util.rotateto(s, xd, dir2=yd, dir3=zd)
    c = s.data['Coordinates']
    pos = np.column_stack([c[:, 1], c[:, 2], c[:, 0]]) * 1e3   # cyclic, det +1
    out = [pos, s.data['Masses'] * C.MASS_TO_MSUN]
    for k in extra:
        out.append(s.data[k])
    del s; gc.collect()
    return out


fig, axes = plt.subplots(2, 1, figsize=(8.4, 11.4), sharex=True, sharey=True)
for row, (sn, when) in enumerate(SNAPS):
    ax = axes[row]
    k = int(np.flatnonzero(SN_ALL == sn)[0])
    t_lo, t_hi = T_ALL[k - 1], T_ALL[k]
    cen, xd, yd, zd = frame(int(sn))

    # --- gas surface density, the background -------------------------------
    gpos, gm = in_frame(int(sn), 0, cen, xd, yd, zd)
    nb = 320
    H, xe, ze = np.histogram2d(gpos[:, 0], gpos[:, 2], bins=nb, weights=gm,
                               range=[[-XLIM, XLIM], [-ZLIM, ZLIM]])
    area = (2 * XLIM / nb) * (2 * ZLIM / nb)
    H = gaussian_filter(H, 1.2) / area                         # Msun / kpc^2
    # The gas fills the frame -- 96 per cent of bins are occupied and the median
    # is only ~100x below the peak -- so a vmin far below the median renders the
    # diffuse halo as shot-noise speckle over everything.  Start the scale at the
    # 75th percentile of the occupied bins instead, and blank what is below it.
    nz = H[H > 0]
    vmin, vmax = np.percentile(nz, 75), np.percentile(nz, 99.9)
    H = np.where(H > vmin, H, np.nan)
    ax.pcolormesh(xe, ze, H.T, cmap=GREY, norm=LogNorm(vmin=vmin, vmax=vmax),
                  rasterized=True, zorder=0)
    print(f'snap {sn}: Sigma_gas scale {vmin:.2e} to {vmax:.2e} Msun/kpc^2')
    del gpos, gm; gc.collect()

    # --- the newborn stars, scattered on top -------------------------------
    spos, _, sids = in_frame(int(sn), 4, cen, xd, yd, zd, extra=('ParticleIDs',))
    sel = (a['tform'] > t_lo) & (a['tform'] <= t_hi) & np.isfinite(a['eps_birth'])
    ids_w, eps_w, zmx_w = a['ids'][sel], a['eps_birth'][sel], zx['zmax_birth'][sel]
    o = np.argsort(sids); ss = sids[o]
    p = np.searchsorted(ss, ids_w)
    ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == ids_w)
    P = spos[o[p[ok]]]; eps, zmx = eps_w[ok], zmx_w[ok]
    disc = (eps > CUT) | (zmx < ZCUT); halo = ~disc
    ax.scatter(P[disc, 0], P[disc, 2], s=1.2, c=cD, alpha=.22, lw=0,
               rasterized=True, zorder=2, label=f'disc-born ({disc.sum():,})')
    ax.scatter(P[halo, 0], P[halo, 2], s=2.6, c=cH, alpha=.55, lw=0,
               rasterized=True, zorder=3, label=f'halo-born ({halo.sum():,})')

    # --- the GS/E itself, top panel only ------------------------------------
    if row == 0:
        pg = np.searchsorted(ss, GSE)
        okg = (pg < len(ss)) & (ss[np.minimum(pg, len(ss) - 1)] == GSE)
        G = spos[o[pg[okg]]]
        OT.density_contours(ax, G[:, 0], G[:, 2], [[-XLIM, XLIM], [-ZLIM, ZLIM]],
                            cG, levels=(0.9, 0.5), bins=70, smooth=1.6, lw=2.2)
        ax.plot([], [], color=cG, lw=2.2, label=f'GS/E debris ({okg.sum():,})')
        print(f'  GS/E centroid (x, z) = ({np.median(G[:, 0]):.1f}, {np.median(G[:, 2]):.1f}) kpc')
    del spos, sids, P; gc.collect()

    ax.set(aspect='equal', xlim=(-XLIM, XLIM), ylim=(-ZLIM, ZLIM), ylabel='$z$ [kpc]')
    ax.text(.025, .965, f'$t = {t_hi:.2f}$ Gyr  —  {when}', transform=ax.transAxes,
            va='top', fontsize=15,
            bbox=dict(fc='white', ec='none', alpha=.78, pad=2.4))
    leg = ax.legend(loc='lower left', markerscale=7, handlelength=1.4,
                    borderpad=.35, labelspacing=.3)
    leg.get_frame().set_alpha(0)
    print(f'snap {sn}: disc-born {disc.sum():,}, halo-born {halo.sum():,}')

axes[1].set_xlabel('$x$ [kpc]')
fig.tight_layout(pad=.5, h_pad=.7)
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/au18_birth_positions_gas.{ext}', bbox_inches='tight')
print(f'\nsaved {OUT}/au18_birth_positions_gas.pdf and .png')
