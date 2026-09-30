"""Publication figure: where the two nitrogen compositions sit, bin by bin in [Fe/H].

The companion to au18_nitrogen_dispersion, which shows that the halo-born
sigma_[N/Fe] belongs to the bridge rather than to the halo orbit.  This shows why:
in every metallicity bin where both compositions are present, they occupy opposite
ends of the lane and have not mixed.

Halo-born stars formed in the snapshot-72 window (t_form = 4.83-4.99 Gyr), one
panel per 0.2 dex bin of [Fe/H], edge-on, coloured by [N/Fe] over the gas surface
density of the same snapshot.  Apertures as in au18_nitrogen_dispersion: violet =
GS/E analogue (within 6 kpc of the clean-debris centroid), black = MW analogue
(within 8 kpc of the host centre).  They are drawn unlabelled here; the companion
figure names them.

Two things about the colour scales, both of which belong in the caption:

- The [N/Fe] scale is DIVERGING ABOUT 0.09, which is the measured divider between
  the two chemical branches, not an arbitrary midpoint.  So blue is the GS/E-like
  composition and red the host-like one, and the pale band is the boundary.
- The gas scale is shared across panels (it is the same snapshot in all of them)
  and is stretched to the disc-to-lane transition, not to the full range of the
  frame, which is dominated by diffuse material.

Marker OPACITY is scaled as 900/N per panel, clipped to 0.20-0.95, because N runs
from 112 to 3,777 across the bins.  At one fixed opacity the crowded intermediate
bins saturate into a solid block and hide the interface between the two streams;
letting markers build up keeps the density contrast instead.

The 150-star threshold of au18_nitrogen_dispersion is NOT applied here.  That cut
exists because a dispersion measured from a hundred stars carries a bootstrap band
wider than the effect; a map of a hundred positions is perfectly legible, and the
sparse metal-poor bin is the one that shows a single composition filling the whole
frame, which is the point.

Nothing is annotated on the panels except the metallicity bin.  The per-panel
counts -- 112, 1,178, 2,435, 3,777, 2,844, 1,419, 637, 184 from the most
metal-poor bin up -- are printed by the script and belong in the caption, where
they also explain why the opacity differs between panels.

Writes Fig_paper/au18_nitrogen_map.pdf and .png.
"""
import gc, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm, ListedColormap
import auriga_public as ap
import config_au18 as C
import au18_frame as AF

OUT = '/data/hz420-2/EoS_explore/Fig_paper'
os.makedirs(OUT, exist_ok=True)

SNAP = 72
CUT, ZCUT = 0.8, 1.5
FBINS = np.arange(-1.2, 0.4 + 1e-9, 0.2)
R_SAT, R_HOST = 6.0, 8.0
DIVIDER, VLO, VHI = 0.09, 0.00, 0.19
W, NGAS = 25.0, 320
NCOL, NROW = 4, 2
RNG = np.random.default_rng(0)
cSAT, cHOST = '#8E24AA', '#2B2B2B'

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


def trunc(name, lo=0., hi=.70):
    return ListedColormap(plt.get_cmap(name)(np.linspace(lo, hi, 256)))


GREY = trunc('Greys')

# --------------------------------- the sample --------------------------------
a = np.load(C.OUT_DIR + '/birth_orbits_actions.npz')
zx = np.load(C.OUT_DIR + '/birth_orbits_zmax.npz')
st = np.load(C.OUT_DIR + '/snapshot_times.npz')
GSE = np.sort(np.load(C.OUT_DIR + '/gse_clean_ids.npy'))
SN_ALL, T_ALL = st['snaps'], st['t_snap']
k = int(np.flatnonzero(SN_ALL == SNAP)[0])
t_lo, t_hi = T_ALL[k - 1], T_ALL[k]

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


def in_frame(ptype, extra=()):
    s = ap.snapshot.load_snapshot(SNAP, ptype, snappath=C.SIM_DIR, verbose=False,
        loadlist=['Coordinates', 'Masses'] + list(extra))
    s = ap.util.CentreOnHalo(s, cen)
    ap.util.rotateto(s, xd, dir2=yd, dir3=zd)
    c = s.data['Coordinates']
    p = np.column_stack([c[:, 1], c[:, 2], c[:, 0]]) * 1e3
    out = [p, s.data['Masses'] * C.MASS_TO_MSUN] + [s.data[e] for e in extra]
    del s; gc.collect()
    return out


spos, _, sids, met = in_frame(4, ('ParticleIDs', 'GFM_Metals'))
# Pin the azimuth so the edge-on view does not foreshorten the GS/E separation.
gcen, _, _, _ = AF.gse_centroid(spos, sids, GSE)
ROT = AF.align_azimuth(gcen)
spos = spos @ ROT.T
g2 = ROT @ gcen

gpos, gm = in_frame(0)
gpos = gpos @ ROT.T
G2, xe, ze = np.histogram2d(gpos[:, 0], gpos[:, 2], bins=NGAS, weights=gm,
                            range=[[-W, W], [-W, W]])
G2 = gaussian_filter(G2, 1.1) / ((2 * W / NGAS) ** 2)
nz = G2[G2 > 0]
gmin, gmax = np.percentile(nz, 55), np.percentile(nz, 99.9)
G2 = np.where(G2 > gmin, G2, np.nan)
del gpos, gm; gc.collect()

sel = (a['tform'] > t_lo) & (a['tform'] <= t_hi) & np.isfinite(a['eps_birth'])
ids_w, eps_w, zmx_w = a['ids'][sel], a['eps_birth'][sel], zx['zmax_birth'][sel]
o = np.argsort(sids); ss = sids[o]
p = np.searchsorted(ss, ids_w)
ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == ids_w)
P, M = spos[o[p[ok]]], met[o[p[ok]]]
eps, zmx = eps_w[ok], zmx_w[ok]
disc = (eps > CUT) | (zmx < ZCUT)
del met, spos; gc.collect()

feh = C.bracket_abundance(M, 'Fe', 'H')
nfe = C.bracket_abundance(M, 'N', 'Fe')
halo = (~disc) & np.isfinite(feh) & np.isfinite(nfe)
print(f'snapshot {SNAP}, t_form {t_lo:.2f}-{t_hi:.2f} Gyr, halo-born {halo.sum():,}')
print(f'gas surface density scale {gmin:.2e} to {gmax:.2e} Msun/kpc^2')

# ---------------------------------- figure -----------------------------------
# Panels are square and abut exactly, so the rects are placed in inches rather
# than left to a gridspec: with wspace = hspace = 0 and aspect='equal', any
# mismatch between the slot shape and the data shape reopens the gaps.
PS = 3.30                                   # panel side, inches
LEFT, BOT, TOP, CBW = .62, .58, .14, 1.62
FW = LEFT + NCOL * PS + CBW
FH = BOT + NROW * PS + TOP
fig = plt.figure(figsize=(FW, FH))
norm = TwoSlopeNorm(vmin=VLO, vcenter=DIVIDER, vmax=VHI)
th = np.linspace(0, 2 * np.pi, 240)
sc = pcm = None

for i in range(len(FBINS) - 1):
    r, c = divmod(i, NCOL)
    ax = fig.add_axes([(LEFT + c * PS) / FW, (BOT + (NROW - 1 - r) * PS) / FH,
                       PS / FW, PS / FH])
    pcm = ax.pcolormesh(xe, ze, G2.T, cmap=GREY,
                        norm=LogNorm(vmin=gmin, vmax=gmax), rasterized=True,
                        zorder=0)
    b = halo & (feh >= FBINS[i]) & (feh < FBINS[i + 1])
    n = int(b.sum())
    j = np.flatnonzero(b)
    RNG.shuffle(j)              # plot order must not bias which colour sits on top
    al = float(np.clip(900. / max(n, 1), .20, .95))
    sc = ax.scatter(P[j, 0], P[j, 2], s=11, c=nfe[j], cmap='RdYlBu_r', norm=norm,
                    lw=0, alpha=al, rasterized=True, zorder=2)
    for r0, c0, x0, z0 in ((R_SAT, cSAT, g2[0], g2[2]), (R_HOST, cHOST, 0., 0.)):
        ax.plot(r0 * np.cos(th) + x0, r0 * np.sin(th) + z0, color=c0, lw=1.8,
                zorder=4)
    ax.set(aspect='equal', xlim=(-W, W), ylim=(-W, W),
           xticks=[-20, -10, 0, 10, 20], yticks=[-20, -10, 0, 10, 20])
    bb = dict(fc='white', ec='none', alpha=.85, pad=2.2)
    ax.text(.045, .955, f'${FBINS[i]:+.1f} < \\mathrm{{[Fe/H]}} < '
                        f'{FBINS[i+1]:+.1f}$', transform=ax.transAxes, va='top',
            fontsize=12.5, bbox=bb, zorder=6)
    if r != NROW - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('$x$ [kpc]')
    if c != 0:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel('$z$ [kpc]')
    print(f'  {FBINS[i]:+.1f} to {FBINS[i+1]:+.1f}: N = {n:>5,}, alpha = {al:.2f}, '
          f'median [N/Fe] = {np.median(nfe[b]) if n else np.nan:+.3f}')

xcb = (LEFT + NCOL * PS + .34) / FW
cb1 = fig.colorbar(sc, cax=fig.add_axes([xcb, (BOT + NROW * PS * .53) / FH,
                                         .17 / FW, NROW * PS * .45 / FH]))
cb1.solids.set_alpha(1.)
cb1.set_label('[N/Fe]', fontsize=14)
cb1.set_ticks([0., .05, .10, .15])
cb2 = fig.colorbar(pcm, cax=fig.add_axes([xcb, (BOT + NROW * PS * .02) / FH,
                                          .17 / FW, NROW * PS * .45 / FH]))
cb2.set_label(r'$\Sigma_{\rm gas}$ [M$_\odot$ kpc$^{-2}$]', fontsize=13)
for cb in (cb1, cb2):
    cb.ax.tick_params(labelsize=11.5, length=3)
    cb.outline.set_linewidth(.8)

for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/au18_nitrogen_map.{ext}', bbox_inches='tight')
print(f'\nsaved {OUT}/au18_nitrogen_map.pdf and .png')
