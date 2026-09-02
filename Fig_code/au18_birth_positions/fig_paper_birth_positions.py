"""Publication figure: where the two birth-orbit populations are born, edge-on.

Two snapshots, side by side in the same disc frame the classification was made in:

  t = 4.99 Gyr  the GS/E pericentre passage, when halo-born formation peaks
  t = 9.41 Gyr  a quiescent late epoch, for contrast

Each shows the stars formed since the previous stored snapshot (~0.15 Gyr), split by

  disc-born   eps > 0.8  OR  z_max < 1.5 kpc
  halo-born   eps <= 0.8 AND z_max >= 1.5 kpc

with eps and z_max from the star's own birth potential (prep_birth_actions.py,
prep_zmax.py).  The point of the figure is that the halo-born class is a genuinely off-plane
population at the merger and a small, sparse one afterwards -- and that it is not
the bar, which is planar and would appear as a thin central line here.

The frame is +-25 kpc in x and +-18 kpc in z, shared by all four panels, with
aspect='equal' so 1 kpc is the same length on both axes and the shapes are not
distorted.  The merger halo-born population extends past it (95th percentile of
|z| = 28 kpc); the fraction outside is annotated on the panel.

NOTE ON THE COLOUR SCALES: each panel is normalised to its own peak, because the
four populations differ in number by more than a factor of twenty (see the N in
each panel) and a shared scale would render the late halo-born panel invisible.
The colour therefore shows the SHAPE of each population, not its abundance; the
abundances are in the annotations and in Fig. au18_birth_orbits.

The colour bars are drawn in units of each panel's own peak for exactly that
reason -- a bar in counts would describe only the panel it sits in, whereas
"fraction of panel peak" is the same scale in all four, so one bar per column is
honest.  They sit inside the bottom row, where those panels are empty.

Writes Fig_paper/au18_birth_positions.pdf and .png.
"""
import gc, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap, LinearSegmentedColormap
import auriga_public as ap
import config_au18 as C

OUT = '/data/hz420-2/EoS_explore/Fig_paper'
os.makedirs(OUT, exist_ok=True)
CUT, ZCUT = 0.8, 1.5
SNAPS = [(72, 'during the merger'), (100, 'after the merger')]
XLIM, ZLIM = 25., 18.

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 13.5, 'axes.labelsize': 15,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12.5,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})

# The halo-born line colour elsewhere in the paper is tomato (#FF6347), but
# matplotlib has no sequential map anchored on it -- `Oranges` runs through a
# yellow-brown that reads as a different population.  Build one that passes
# through tomato at its midpoint, so a dense halo-born pixel here is the same
# hue as the halo-born curve in au18_birth_orbits.
# Node positions matter as much as the colours: spacing them evenly puts tomato
# at the midpoint, which leaves most of a dense panel in the saturated half and
# reads as blood-red.  Tomato sits at 0.72 instead, so the bulk of the map is the
# pale half and only genuinely dense bins darken past it.
TOMATO = LinearSegmentedColormap.from_list('tomato_seq', [
    (0.00, '#FFF5F2'), (0.25, '#FFDCD1'), (0.50, '#FFB29E'),
    (0.72, '#FF6347'), (0.88, '#DD4429'), (1.00, '#A82915')])


def trunc(cmap, lo=.22, hi=1.):
    """Drop the palest end of a sequential map.

    With LogNorm(vmin=1) a bin holding a single star sits at the very bottom of
    the colour range, which for Blues/Reds is almost white -- so a sparse
    population (the 1,214 late halo-born stars) all but disappears.  Starting the
    map partway in keeps single-star bins visible.

    Takes either a registered name or a Colormap object (TOMATO is not registered).
    """
    cm = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    return ListedColormap(cm(np.linspace(lo, hi, 256)))


a = np.load(C.OUT_DIR + '/birth_orbits_actions.npz')
zx = np.load(C.OUT_DIR + '/birth_orbits_zmax.npz')
assert np.array_equal(a['ids'], zx['ids'])
st = np.load(C.OUT_DIR + '/snapshot_times.npz')
SN_ALL, T_ALL = st['snaps'], st['t_snap']


def stars_in_frame(sn):
    """Star positions in that snapshot's own disc frame, same convention as prep."""
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
    s = ap.snapshot.load_snapshot(sn, 4, snappath=C.SIM_DIR, verbose=False,
        loadlist=['ParticleIDs', 'Coordinates', 'Masses', 'Velocities',
                  'GFM_StellarFormationTime'])
    s = ap.util.CentreOnHalo(s, cen)
    ap.util.rotateto(s, xd, dir2=yd, dir3=zd)
    real = s.data['GFM_StellarFormationTime'] > 0
    c = s.data['Coordinates'][real]
    # cyclic permutation (0,1,2) -> (1,2,0): determinant +1, matches the prep step
    pos = np.column_stack([c[:, 1], c[:, 2], c[:, 0]]) * 1e3
    ids = s.data['ParticleIDs'][real]
    del s, c; gc.collect()
    return ids, pos


fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.4), sharex=True, sharey=True)
info = []
for row, (sn, when) in enumerate(SNAPS):
    k = int(np.flatnonzero(SN_ALL == sn)[0])
    t_lo, t_hi = T_ALL[k - 1], T_ALL[k]
    sel = (a['tform'] > t_lo) & (a['tform'] <= t_hi) & np.isfinite(a['eps_birth'])
    ids_w = a['ids'][sel]
    eps_w, zmx_w = a['eps_birth'][sel], zx['zmax_birth'][sel]
    ids, pos = stars_in_frame(sn)
    o = np.argsort(ids); ss = ids[o]
    p = np.searchsorted(ss, ids_w)
    ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == ids_w)
    P = pos[o[p[ok]]]; eps, zmx = eps_w[ok], zmx_w[ok]
    disc = (eps > CUT) | (zmx < ZCUT)

    for col, (lab, m, cmap, nb) in enumerate(
            # Coarser pixels for the halo-born column: those populations are far
            # sparser (1,214 stars in the late panel against 26,221 beside it), so
            # at the disc-born pixel size most of their bins hold a single star and
            # the map reads as noise rather than as a distribution.
            [('disc-born', disc, trunc('Blues', .14, .90), 200),
             ('halo-born', ~disc, trunc(TOMATO, .14, .90), 140)]):
        ax = axes[row, col]
        h, xe, ze = np.histogram2d(P[m, 0], P[m, 2], bins=[nb, nb],
                                   range=[[-XLIM, XLIM], [-ZLIM, ZLIM]])
        h = h / np.nanmax(h)                      # -> fraction of this panel's peak
        pcm = ax.pcolormesh(xe, ze, h.T, cmap=cmap, norm=LogNorm(vmin=1e-3, vmax=1.),
                            rasterized=True)
        ax.axhline(0, color='.6', lw=.6, zorder=0)
        ax.set(aspect='equal', xlim=(-XLIM, XLIM), ylim=(-ZLIM, ZLIM))
        mz = np.median(np.abs(P[m, 2]))
        out = 100 * np.mean((np.abs(P[m, 2]) > ZLIM) | (np.abs(P[m, 0]) > XLIM))
        # The merger halo-born population fills its panel, so the annotations need
        # their own background to stay legible over it.
        bb = dict(fc='white', ec='none', alpha=.72, pad=1.6)
        ax.text(.035, .95, f'N = {m.sum():,}', transform=ax.transAxes, va='top',
                fontsize=13, bbox=bb)
        ax.text(.035, .875, f'median $|z|$ = {mz:.2f} kpc', transform=ax.transAxes,
                va='top', fontsize=12, bbox=bb)
        # The merger halo-born population reaches beyond the frame; say so rather
        # than letting the crop imply it stops there.
        if out > 1.:
            ax.text(.035, .80, f'{out:.0f}% beyond the frame', transform=ax.transAxes,
                    va='top', fontsize=11, color='.30', bbox=bb)
        if row == 0:
            ax.text(.5, 1.04, lab, transform=ax.transAxes, ha='center', fontsize=16)
        if col == 1:
            ax.text(1.035, .5, f'$t = {t_hi:.2f}$ Gyr\n{when}', transform=ax.transAxes,
                    rotation=270, va='center', ha='left', fontsize=14.5)
        if row == 1:
            cax = ax.inset_axes([.09, .085, .56, .030])
            cb = fig.colorbar(pcm, cax=cax, orientation='horizontal',
                              ticks=[1e-3, 1e-2, 1e-1, 1.])
            cb.ax.set_xticklabels(['0.001', '0.01', '0.1', '1'])
            cb.ax.tick_params(labelsize=10.5, length=2.5, pad=1.5)
            cb.outline.set_linewidth(.7)
            cax.set_title('density / panel peak', fontsize=10.5, pad=3.5)
        info.append((sn, lab, m.sum(), mz, out))
    del ids, pos, P; gc.collect()

for ax in axes[1, :]:
    ax.set_xlabel('$x$ [kpc]')
for ax in axes[:, 0]:
    ax.set_ylabel('$z$ [kpc]')
fig.tight_layout(pad=.5, w_pad=.6, h_pad=.6)
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/au18_birth_positions.{ext}', bbox_inches='tight')
for sn, lab, n, mz, out in info:
    print(f'snap {sn:3d}  {lab:10s} N={n:>7,}  median |z| = {mz:5.2f} kpc  '
          f'outside the frame: {out:.2f}%')
print(f'\nsaved {OUT}/au18_birth_positions.pdf and .png')
