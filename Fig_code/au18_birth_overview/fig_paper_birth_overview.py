"""Publication figure: where the two birth classes form, and when -- one row.

The 1x3 variant of au18_birth_classes.  Its two maps are merged into a single
panel with both classes overplotted, in the style of au18_birth_positions_gas,
and the star-formation panels follow to its right.

  (a)  the stars formed in the snapshot-72 window (t_form = 4.83-4.99 Gyr), at
       the GS/E pericentre, edge-on over that snapshot's gas surface density,
       both birth classes on one map, with contours enclosing 50 and 90 per cent
       of the clean GS/E debris
  (b)  the star-formation history split by the same classes
  (c)  their ratio

  disc-born   eps > 0.8  OR  z_max < 1.5 kpc
  halo-born   eps <= 0.8 AND z_max >= 1.5 kpc

au18_birth_classes is kept; this is an addition, not a replacement.  The merged
map trades the clean separation of the two-panel version -- where each class had
the gas to itself -- for a direct comparison of where they sit relative to one
another.  Draw order is disc-born first, halo-born over it, so the off-plane
class is not buried under the disc; with the reverse order the halo-born lane is
invisible where it crosses the midplane.

The panel rectangles are placed explicitly in inches.  Panel (a) is forced to
aspect='equal' and the others are not, and tight_layout will not keep a mixed row
aligned -- it shrinks the equal-aspect axes inside its slot and centres it.

No counts are annotated: they belong in the caption, and the script prints them.
Opacity is scaled to each class's N so the 12,608-star class does not vanish
against the 26,064-star one; both are scatter rather than a binned density so the
gas stays visible everywhere.

Writes Fig_paper/au18_birth_overview.pdf and .png.
"""
import gc, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
import auriga_public as ap
import config_au18 as C
import au18_frame as AF
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import orbit_tools as OT

OUT = '/data/hz420-2/EoS_explore/Fig_paper'
os.makedirs(OUT, exist_ok=True)
SNAP = 72
CUT, ZCUT = 0.8, 1.5
XLIM, ZLIM = 25., 18.                    # widened below if the GS/E lies outside
BW_T = 0.08                              # SFR kernel sigma in Gyr, not a bin width
T_PERI, T_SPIN = 5.0, 3.4
cD, cH, cT, cM, cS = '#1F6FB2', '#FF6347', '#2B2B2B', '#8E24AA', '#00897B'

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    # house sizes x 1.1; these panels are reproduced small and 15 pt reads thin
    'font.size': 15, 'axes.labelsize': 16.5,
    'xtick.labelsize': 14.5, 'ytick.labelsize': 14.5, 'legend.fontsize': 13.5,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})


def trunc(name, lo=0., hi=.70):
    return ListedColormap(plt.get_cmap(name)(np.linspace(lo, hi, 256)))


GREY = trunc('Greys')

# ----------------------------- the birth orbits ------------------------------
a = np.load(C.OUT_DIR + '/birth_orbits_actions.npz')
zx = np.load(C.OUT_DIR + '/birth_orbits_zmax.npz')
q = np.load(C.OUT_DIR + '/insitu_imass.npz')
st_ = np.load(C.OUT_DIR + '/snapshot_times.npz')
GSE = np.sort(np.load(C.OUT_DIR + '/gse_clean_ids.npy'))
k = int(np.flatnonzero(st_['snaps'] == SNAP)[0])
t_lo, t_hi = float(st_['t_snap'][k - 1]), float(st_['t_snap'][k])

tf, eb, mi, zm = a['tform'], a['eps_birth'], q['imass'], zx['zmax_birth']
gf = np.isfinite(eb) & np.isfinite(mi) & np.isfinite(zm)
tfg, ebg, mig, zmg = tf[gf], eb[gf], mi[gf], zm[gf]
disc_all = (ebg > CUT) | (zmg < ZCUT)
TMIN = np.floor(tfg.min() * 10) / 10
print(f'{gf.sum():,} in-situ stars with a birth orbit; '
      f'halo-born {100 * (~disc_all).mean():.1f} per cent')

# ------------------------------- the snapshot --------------------------------
sub = ap.subhalos.subfind(SNAP, directory=C.SIM_DIR,
                          loadlist=['SubhaloPos', 'Group_R_Crit200'])
r200 = float(sub.data['Group_R_Crit200'][0]); cen = sub.data['SubhaloPos'][0]
ref = ap.snapshot.load_snapshot(SNAP, 4, snappath=C.SIM_DIR, verbose=False,
    loadlist=['Coordinates', 'Masses', 'Potential', 'Velocities'])
ref = ap.util.CentreOnHalo(ref, cen)
ref = ap.util.apply_mask(ref, stars=False, radialcut=.5 * r200)
ist, = np.where(ap.util.r(ref) < .1 * r200)
L = np.cross(ref.data['Coordinates'][ist],
             ref.data['Velocities'][ist] * ref.data['Masses'][ist, None])
Ld = L.sum(0); Ld /= np.sqrt((Ld ** 2).sum())
xd, yd, zd = ap.util.get_principal_axis(ref, ist, L=Ld)
del ref; gc.collect()


def in_frame(ptype, rot=None, extra=()):
    s = ap.snapshot.load_snapshot(SNAP, ptype, snappath=C.SIM_DIR, verbose=False,
        loadlist=['Coordinates', 'Masses'] + list(extra))
    s = ap.util.CentreOnHalo(s, cen)
    ap.util.rotateto(s, xd, dir2=yd, dir3=zd)
    c = s.data['Coordinates']
    p = np.column_stack([c[:, 1], c[:, 2], c[:, 0]]) * 1e3
    if rot is not None:
        p = p @ rot.T
    out = [p, s.data['Masses'] * C.MASS_TO_MSUN] + [s.data[e] for e in extra]
    del s; gc.collect()
    return out


sp0, _, sid0 = in_frame(4, extra=('ParticleIDs',))
gcen, _, _, _ = AF.gse_centroid(sp0, sid0, GSE)
# Pin the azimuth: the rotation about the disc axis is otherwise arbitrary and
# would foreshorten the host-satellite separation.
ROT = AF.align_azimuth(gcen)
g2 = ROT @ gcen
XLIM = max(XLIM, abs(g2[0]) + 6.); ZLIM = max(ZLIM, abs(g2[2]) + 5.)
del sp0, sid0; gc.collect()

gpos, gm = in_frame(0, rot=ROT)
nb = 300
G2, xe, ze = np.histogram2d(gpos[:, 0], gpos[:, 2], bins=nb, weights=gm,
                            range=[[-XLIM, XLIM], [-ZLIM, ZLIM]])
G2 = gaussian_filter(G2, 1.1) / ((2 * XLIM / nb) * (2 * ZLIM / nb))
nz = G2[G2 > 0]
gmin, gmax = np.percentile(nz, 55), np.percentile(nz, 99.9)
G2 = np.where(G2 > gmin, G2, np.nan)
del gpos, gm; gc.collect()

spos, _, sids = in_frame(4, rot=ROT, extra=('ParticleIDs',))
sel = (a['tform'] > t_lo) & (a['tform'] <= t_hi) & np.isfinite(a['eps_birth'])
ids_w, eps_w, zmx_w = a['ids'][sel], a['eps_birth'][sel], zx['zmax_birth'][sel]
o = np.argsort(sids); ss = sids[o]
p = np.searchsorted(ss, ids_w)
ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == ids_w)
P = spos[o[p[ok]]]
eps, zmx = eps_w[ok], zmx_w[ok]
disc = (eps > CUT) | (zmx < ZCUT)
pg = np.searchsorted(ss, GSE)
okg = (pg < len(ss)) & (ss[np.minimum(pg, len(ss) - 1)] == GSE)
Gs = spos[o[pg[okg]]]
del spos, sids; gc.collect()
print(f'snapshot {SNAP} (t = {t_hi:.2f} Gyr): disc-born {disc.sum():,}, '
      f'halo-born {(~disc).sum():,}; gas scale {gmin:.2e}-{gmax:.2e}')

# ---------------------------------- layout -----------------------------------
PH = 3.95                                     # the shared panel height, inches
PWA = PH * XLIM / ZLIM                        # panel (a): forced equal aspect
PWB = 4.75                                    # the two star-formation panels
LEFT, RIGHT, BOT, TOP, WG = .80, .20, .66, .22, .98
FW = LEFT + PWA + WG + 2 * PWB + WG + RIGHT
FH = BOT + PH + TOP
fig = plt.figure(figsize=(FW, FH))
x0 = LEFT / FW
AXT = [fig.add_axes([x0, BOT / FH, PWA / FW, PH / FH])]
AXB = [fig.add_axes([(LEFT + PWA + WG + i * (PWB + WG)) / FW, BOT / FH,
                     PWB / FW, PH / FH]) for i in range(2)]

# ---------------------------------- (a) --------------------------------------
ax = AXT[0]
pcm = ax.pcolormesh(xe, ze, G2.T, cmap=GREY, norm=LogNorm(vmin=gmin, vmax=gmax),
                    rasterized=True, zorder=0)
# disc-born first, halo-born over it: with the reverse order the halo-born lane
# is buried wherever it crosses the midplane
for lab, m, colr, ms, zo in (('disc-born', disc, cD, 1.6, 2),
                             ('halo-born', ~disc, cH, 3.0, 3)):
    al = float(np.clip(2500. / max(m.sum(), 1), .10, .75))
    ax.scatter(P[m, 0], P[m, 2], s=ms, c=colr, alpha=al, lw=0, rasterized=True,
               zorder=zo)
    # a proxy at full opacity: a legend handle taken from the scatter inherits
    # its alpha, which is 0.1-0.75 here and leaves the key unreadable
    ax.scatter([], [], s=46, c=colr, alpha=1., lw=0, label=lab)
OT.density_contours(ax, Gs[:, 0], Gs[:, 2], [[-XLIM, XLIM], [-ZLIM, ZLIM]],
                    cM, levels=(0.9, 0.5), bins=70, smooth=1.6, lw=2.0)
ax.plot([], [], color=cM, lw=2.0, label='GS/E debris')
bb = dict(fc='white', ec='none', alpha=.80, pad=1.8)
ax.text(.03, .955, f'$t$ = {t_hi:.2f} Gyr, at the GS/E pericentre',
        transform=ax.transAxes, va='top', fontsize=13.5, bbox=bb)
# the tag goes to the lower left, the only corner left once the epoch label has
# the upper left and the legend the upper right; it clears the GS/E contours,
# whose leftmost extent is x = -19 kpc
ax.text(.03, .045, '(a)', transform=ax.transAxes, va='bottom',
        fontsize=18, fontweight='bold', bbox=bb)
ax.set(aspect='equal', xlim=(-XLIM, XLIM), ylim=(-ZLIM, ZLIM),
       xlabel='$x$ [kpc]', ylabel='$z$ [kpc]')
# upper right: the one corner of this frame with neither the disc, the lane nor
# the GS/E debris in it
leg = ax.legend(loc='upper right', handlelength=1.4, borderpad=.3,
                labelspacing=.35, fontsize=12.5)
leg.get_frame().set_alpha(0)
cax = ax.inset_axes([.44, .075, .52, .030])
cb = fig.colorbar(pcm, cax=cax, orientation='horizontal')
cb.ax.tick_params(labelsize=10.5, length=2.5, pad=1.5)
cb.outline.set_linewidth(.7)
cax.set_title(r'$\Sigma_{\rm gas}$ [M$_\odot$ kpc$^{-2}$]', fontsize=11.5, pad=9)
cax.title.set_bbox(bb)

# ------------------------------ (c) and (d) ----------------------------------
def sfr_kde(t, w, grid, bw=BW_T):
    """SFR [Msun/yr] as a Gaussian kernel density in cosmic time.

    bw is a Gaussian SIGMA, not a bin width.  sigma = 0.15 would have FWHM
    0.35 Gyr and oversmooth the narrow halo-born spike while leaving the broad
    disc-born curve alone, dragging the ratio in (d) down by a third.
    """
    fine = np.arange(grid[0] - 8 * bw, grid[-1] + 8 * bw, bw / 8.)
    h = np.histogram(t, bins=fine, weights=w)[0]
    sm = gaussian_filter1d(h, 8.) / (bw / 8.)
    ctr = .5 * (fine[:-1] + fine[1:])
    return np.interp(grid, ctr, sm) / 1e9


def markers(ax, lab=False):
    ax.axvline(T_SPIN, color=cS, ls=':', lw=2.6, zorder=1,
               label='disc spin-up' if lab else None)
    ax.axvline(T_PERI, color=cM, ls='--', lw=2.4, zorder=1,
               label='GS/E pericentre' if lab else None)


tgrid = np.linspace(TMIN, C.T0_GYR, 500)
sT = sfr_kde(tfg, mig, tgrid)
sD = sfr_kde(tfg[disc_all], mig[disc_all], tgrid)
sH = sfr_kde(tfg[~disc_all], mig[~disc_all], tgrid)

ax = AXB[0]
markers(ax, lab=True)
ax.plot(tgrid, sT, color=cT, lw=1.8, label='total')
ax.plot(tgrid, sD, color=cD, lw=1.8, label='disc-born')
ax.fill_between(tgrid, 0, sD, color=cD, alpha=.15, lw=0)
ax.plot(tgrid, sH, color=cH, lw=1.8, label='halo-born')
ax.fill_between(tgrid, 0, sH, color=cH, alpha=.22, lw=0)
ax.set(xlim=(TMIN, C.T0_GYR), ylim=(0, 1.06 * sT.max()), xlabel='cosmic time [Gyr]',
       ylabel=r'SFR [M$_\odot$ yr$^{-1}$]', xticks=np.arange(2, 14, 2))
ax.legend(loc='upper right', handlelength=1.6, borderpad=.25,
          labelspacing=.35, ncol=2, fontsize=12.5)
ax.text(.035, .945, '(b)', transform=ax.transAxes, va='top', fontsize=18,
        fontweight='bold')

ax = AXB[1]
rat = np.divide(sH, sD, out=np.full_like(sH, np.nan), where=sD > 1.)
markers(ax)
ax.plot(tgrid, rat, color=cH, lw=2.2)
# the full range is kept: the rise before t ~ 2.5 Gyr is the pre-disc era, when
# there is barely a disc to normalise against, and clipping it would hide that
ax.set(xlim=(TMIN, C.T0_GYR), ylim=(0, 1.05 * np.nanmax(rat)),
       xlabel='cosmic time [Gyr]', ylabel='halo-born / disc-born SFR',
       xticks=np.arange(2, 14, 2))
ax.text(.035, .945, '(c)', transform=ax.transAxes, va='top', fontsize=18,
        fontweight='bold')

pk = np.nanargmax(np.where(tgrid > 2.5, rat, np.nan))
print(f'peak halo/disc ratio {rat[pk]:.2f} at t = {tgrid[pk]:.2f} Gyr; '
      f'halo-born SFR there {sH[pk]:.2f}, total {sT[pk]:.2f} Msun/yr')
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/au18_birth_overview.{ext}', bbox_inches='tight')
print(f'\nsaved {OUT}/au18_birth_overview.pdf and .png')
