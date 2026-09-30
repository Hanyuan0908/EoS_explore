"""Publication figure: the nitrogen dispersion of the two birth classes, and
where it comes from.

Stars formed in the snapshot-72 window (t_form = 4.83-4.99 Gyr), at the GS/E
pericentre passage, split by the birth orbit they were put on:

  disc-born   eps > 0.8  OR  z_max < 1.5 kpc
  halo-born   eps <= 0.8 AND z_max >= 1.5 kpc

(a) the halo-born stars in the edge-on plane, coloured by [N/Fe], with the two
    apertures used in (c): the GS/E analogue (within R_SAT of the clean-debris
    centroid) and the MW analogue (within R_HOST of the host centre).
(b) sigma_[N/Fe] against [Fe/H] in fixed bins, disc-born against halo-born.
(c) the halo-born curve split by birth site: inside the GS/E analogue, inside the
    MW analogue, and along the bridge between them.

Fixed [Fe/H] bins matter: the two classes differ by 0.61 dex in median
metallicity and sigma runs with [Fe/H], so an unbinned comparison returns that
gradient as if it were a difference between the classes.

The result the figure carries: the dispersion is a property of the BRIDGE, not of
the halo orbit.  Halo-born stars formed inside either compact aperture sit near
0.016 dex at every metallicity; only the ones formed in the lane between the two
galaxies reach 0.057.  The bridge is where GS/E gas and host gas are still
unmixed, and stars there sample one or the other -- panel (a) shows the two
compositions occupying opposite ends of the lane.

CAVEAT for the caption: Auriga's abundances are not the Milky Way's.  Every
dispersion here is 0.01-0.06 dex, against the ~0.3 dex that separates the MW's
two alpha sequences.  Read them as ISM inhomogeneity within the model, not
against an observed sigma_[N/Fe].

Writes Fig_paper/au18_nitrogen_dispersion.pdf and .png.
"""
import gc, os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import auriga_public as ap
import config_au18 as C
import au18_frame as AF

OUT = '/data/hz420-2/EoS_explore/Fig_paper'
os.makedirs(OUT, exist_ok=True)

SNAP = 72
CUT, ZCUT = 0.8, 1.5
FBINS = np.arange(-1.2, 0.4 + 1e-9, 0.2)
# a bin is drawn only if it holds NMIN stars: below that the bootstrap band
# is wider than the differences the figure is about
NMIN, NSITE, NBOOT = 150, 150, 500
R_SAT, R_HOST = 6.0, 8.0
DIVIDER = 0.09              # measured branch divider; the pale band of the map
VLO, VHI = 0.00, 0.19
W = 25.0
RNG = np.random.default_rng(0)

cDISC, cHALO = '#1F6FB2', '#FF6347'       # born-cold / born-hot house colours
cSAT, cBR, cHOST = '#8E24AA', '#00897B', '#2B2B2B'

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    # House sizes x 1.26: at the width these figures are reproduced, the 15 pt
    # house axis label renders small on the page.  Scaling every size by one
    # factor keeps the internal proportions unchanged.
    'font.size': 17, 'axes.labelsize': 18.5,
    'xtick.labelsize': 16.5, 'ytick.labelsize': 16.5, 'legend.fontsize': 16,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})
ctr = .5 * (FBINS[:-1] + FBINS[1:])
NB = len(ctr)


def mad_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def disp_profile(feh, y, m, nmin):
    """Robust sigma(y) per [Fe/H] bin, with the 16-84 range of NBOOT bootstraps."""
    s, lo, hi = (np.full(NB, np.nan) for _ in range(3))
    cnt = np.zeros(NB, int)
    ok = m & np.isfinite(feh) & np.isfinite(y)
    for k in range(NB):
        b = ok & (feh >= FBINS[k]) & (feh < FBINS[k + 1])
        cnt[k] = b.sum()
        if cnt[k] < nmin:
            continue
        v = y[b]
        s[k] = mad_sigma(v)
        bs = np.array([mad_sigma(v[RNG.integers(0, len(v), len(v))])
                       for _ in range(NBOOT)])
        lo[k], hi[k] = np.percentile(bs, (16, 84))
    return s, lo, hi, cnt


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

s = ap.snapshot.load_snapshot(SNAP, 4, snappath=C.SIM_DIR, verbose=False,
    loadlist=['Coordinates', 'ParticleIDs', 'GFM_Metals'])
s = ap.util.CentreOnHalo(s, cen)
ap.util.rotateto(s, xd, dir2=yd, dir3=zd)
cc = s.data['Coordinates']
pos = np.column_stack([cc[:, 1], cc[:, 2], cc[:, 0]]) * 1e3
sids, met = s.data['ParticleIDs'], s.data['GFM_Metals']
del s; gc.collect()

# Pin the azimuth so the edge-on view does not foreshorten the GS/E separation.
gcen, _, _, _ = AF.gse_centroid(pos, sids, GSE)
ROT = AF.align_azimuth(gcen)
pos = pos @ ROT.T
g2 = ROT @ gcen

sel = (a['tform'] > t_lo) & (a['tform'] <= t_hi) & np.isfinite(a['eps_birth'])
ids_w, eps_w, zmx_w = a['ids'][sel], a['eps_birth'][sel], zx['zmax_birth'][sel]
o = np.argsort(sids); ss = sids[o]
p = np.searchsorted(ss, ids_w)
ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == ids_w)
P, M = pos[o[p[ok]]], met[o[p[ok]]]
eps, zmx = eps_w[ok], zmx_w[ok]
disc = (eps > CUT) | (zmx < ZCUT)
del met, pos; gc.collect()

feh = C.bracket_abundance(M, 'Fe', 'H')
nfe = C.bracket_abundance(M, 'N', 'Fe')
halo = (~disc) & np.isfinite(feh) & np.isfinite(nfe)
d_gse = np.linalg.norm(P - g2, axis=1)
d_host = np.linalg.norm(P, axis=1)
sat = d_gse < R_SAT
host = (~sat) & (d_host < R_HOST)
bridge = ~(sat | host)
SITES = [('within GS/E-analogue', halo & sat, cSAT),
         ('along the bridge', halo & bridge, cBR),
         ('within MW-analogue', halo & host, cHOST)]

print(f'snapshot {SNAP}, t_form {t_lo:.2f}-{t_hi:.2f} Gyr')
print(f'  disc-born {disc.sum():,}   halo-born {halo.sum():,}')
print(f'  GS/E centroid (x,z) = ({g2[0]:.1f}, {g2[2]:.1f}) kpc, '
      f'|r| = {np.linalg.norm(g2):.1f} kpc')
for lab, m, _ in SITES:
    print(f'  {lab:14s} {m.sum():>6,}  ({100*m.sum()/halo.sum():>4.1f} % of halo-born)  '
          f'median [Fe/H] = {np.nanmedian(feh[m]):+.3f}, '
          f'sigma_[N/Fe] = {mad_sigma(nfe[m]):.4f}')

# ---------------------------------- figure -----------------------------------
# Panels (b) and (c) are 6:4 and share an axes height with (a), which is square
# because it is forced to aspect='equal' over a 50 x 50 kpc frame.  The rects are
# placed explicitly, in inches converted to figure fraction, because tight_layout
# will not honour a mixed-aspect row.
FW, FH = 17.0, 4.5
LEFT, BOT, AXH, GAP = .55, .80, 3.65, .85
WA, WBC = AXH, AXH * 6. / 4.
fig = plt.figure(figsize=(FW, FH))
x0 = LEFT / FW
rects = [(x0, WA), (x0 + (WA + GAP) / FW, WBC),
         (x0 + (WA + 2 * GAP + WBC) / FW, WBC)]
AX = [fig.add_axes([x, BOT / FH, w / FW, AXH / FH]) for x, w in rects]

# ---- (a) the map
ax = AX[0]
ax.scatter(P[disc, 0], P[disc, 2], s=1.4, c='0.90', lw=0, rasterized=True, zorder=0)
j = np.flatnonzero(halo)
RNG.shuffle(j)                      # plot order must not bias which colour is on top
sc = ax.scatter(P[j, 0], P[j, 2], s=7.5, c=nfe[j], cmap='RdYlBu_r',
                norm=TwoSlopeNorm(vmin=VLO, vcenter=DIVIDER, vmax=VHI),
                lw=0, alpha=.35, rasterized=True, zorder=2)
th = np.linspace(0, 2 * np.pi, 240)
for r0, c0, x0, z0 in ((R_SAT, cSAT, g2[0], g2[2]), (R_HOST, cHOST, 0., 0.)):
    ax.plot(r0 * np.cos(th) + x0, r0 * np.sin(th) + z0, color=c0, lw=2.0, zorder=4)
bb = dict(fc='white', ec='none', alpha=.85, pad=1.8)
ax.text(-13.5, -22.3, 'GS/E analogue', color=cSAT, ha='center', va='center',
        fontsize=16, bbox=bb, zorder=5)
ax.text(0., R_HOST + 1.4, 'MW analogue', color=cHOST, ha='center', va='bottom',
        fontsize=16, bbox=bb, zorder=5)
# led out to the empty lower right rather than sitting on the lane it names
ax.annotate('bridge', xy=(-5.5, -6.8), xytext=(13.5, -17.5), color=cBR,
            ha='center', va='center', fontsize=16, bbox=bb, zorder=5,
            arrowprops=dict(arrowstyle='-', color=cBR, lw=1.3,
                            shrinkA=2, shrinkB=2))
ax.set(aspect='equal', xlim=(-W, W), ylim=(-W, W),
       xlabel='$x$ [kpc]', ylabel='$z$ [kpc]')
cax = ax.inset_axes([.545, .885, .41, .032])
cb = fig.colorbar(sc, cax=cax, orientation='horizontal')
cb.solids.set_alpha(1.)
cb.set_ticks([0., .05, .10, .15])
cb.ax.tick_params(labelsize=13, length=2.5, pad=1.5)
cb.outline.set_linewidth(.7)
cax.set_title('[N/Fe]', fontsize=14, pad=3)
cax.title.set_bbox(bb)
ax.text(.035, .965, '(a)', transform=ax.transAxes, va='top', fontsize=19.5,
        fontweight='bold')

# ---- (b) the two birth classes
ax = AX[1]
for lab, m, col in (('disc-born', disc & np.isfinite(nfe), cDISC),
                    ('halo-born', halo, cHALO)):
    sg, lo, hi, cnt = disp_profile(feh, nfe, m, NMIN)
    g = np.isfinite(sg)
    ax.plot(ctr[g], sg[g], color=col, lw=2.8, label=lab)
    ax.fill_between(ctr[g], lo[g], hi[g], color=col, alpha=.22, lw=0, rasterized=True)
    print(f'  (b) {lab:10s} ' + '  '.join(f'{v:.4f}' if np.isfinite(v) else '  --  '
                                          for v in sg))
ax.set(xlabel='[Fe/H]', ylabel=r'$\sigma_{\rm [N/Fe]}$ [dex]',
       xlim=(-1.05, .45), ylim=(.005, .07))
ax.legend(loc='upper right', handlelength=1.6)
ax.text(.035, .965, '(b)', transform=ax.transAxes, va='top', fontsize=19.5,
        fontweight='bold')

# ---- (c) the halo-born class split by birth site
ax = AX[2]
sg_h = disp_profile(feh, nfe, halo, NMIN)[0]
gh = np.isfinite(sg_h)
ax.plot(ctr[gh], sg_h[gh], color=cHALO, lw=2.8, alpha=.45, zorder=1,
        label='halo-born, all')
for lab, m, col in SITES:
    sg, lo, hi, cnt = disp_profile(feh, nfe, m, NSITE)
    g = np.isfinite(sg)
    ax.plot(ctr[g], sg[g], color=col, lw=2.4, ls='--', zorder=3, label=lab)
    ax.fill_between(ctr[g], lo[g], hi[g], color=col, alpha=.20, lw=0, rasterized=True)
    print(f'  (c) {lab:14s} ' + '  '.join(f'{v:.4f}' if np.isfinite(v) else '  --  '
                                          for v in sg))
ax.set(xlabel='[Fe/H]', ylabel=r'$\sigma_{\rm [N/Fe]}$ [dex]',
       xlim=(-1.05, .45), ylim=(.005, .07))
ax.legend(loc='upper right', handlelength=1.5, fontsize=13.5,
          labelspacing=.3, borderaxespad=.4)
ax.text(.035, .965, '(c)', transform=ax.transAxes, va='top', fontsize=19.5,
        fontweight='bold')

for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/au18_nitrogen_dispersion.{ext}', bbox_inches='tight')
print(f'\nsaved {OUT}/au18_nitrogen_dispersion.pdf and .png')
