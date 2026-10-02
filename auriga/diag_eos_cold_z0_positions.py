"""Diagnostic: where the born-cold Eos-like stars sit TODAY.

The born-cold channel is the half of the Eos selection that was on a disc orbit at
birth (v_phi,birth >= 150 km/s) and is on a slow, eccentric orbit now -- i.e. the
heated-disc half.  au18_eos_rbirth_rapo shows where they were born and how far out
they reach; this shows the z = 0 spatial distribution itself.

  Eos-like   |v_phi| < 80 km/s AND ecc > 0.6 at z = 0, among stars formed in
             t_form = 4.99-6.54 Gyr   (the selection of au18_vr_vphi_three)
  born-cold  v_phi,birth >= 150 km/s

Two panels in the z = 0 disc frame: face-on and edge-on, on a common x axis so
the flattening can be read off directly.  The full merger-window sample is drawn
underneath in grey, so "where the born-cold stars are" can be compared against
"where the stars of that epoch are" rather than against nothing.

FRAME.  util.align_galaxy puts the disc angular momentum on component 0, so the
disc plane is components (1, 2).  The cyclic mapping (1, 2, 0) -> (x, y, z) is
used here, determinant +1; the transposition (2, 1, 0) is a reflection and would
silently negate L_z (see CONVENTIONS.md).

    usage:  diag_eos_cold_z0_positions.py [R_MIN]        (default 0)

R_MIN applies a cylindrical cut R > R_MIN kpc at z = 0, to step outside the bar.
The face-on panel of the uncut version is visibly barred -- elongated and bilobed
with overdensities a few kpc either side of the centre -- and that structure, not
the Eos selection, dominates the inner few kpc.

ALREADY IN THE DATA, and not optional: merger_birth_vs_z0_kinematics.npz carries
a spherical aperture 3 < r < 30 kpc.  The minimum kept radius is exactly 3.000
kpc and 0 per cent of stars at r = 2.5-3.0 survive.  That aperture removes 49 per
cent of the in-situ stars formed in this window -- 164,436 of 336,262 -- and they
are overwhelmingly central (median r = 1.37 kpc against 7.35 kpc for the kept
ones).  So the inner edge of every panel here is an aperture, not a physical
feature, and R_MIN only moves it further out.

Writes figures/au18_eos_cold_z0_positions[_R<N>].png
"""
import gc, os, sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod, util

OUT = C.FIG_DIR
os.makedirs(OUT, exist_ok=True)
R_MIN = float(sys.argv[1]) if len(sys.argv) > 1 else 0.
TAG = '' if R_MIN <= 0 else f'_R{R_MIN:g}'
VPHI_MAX, ECC_MIN, VPHI_SPLIT = 80., 0.6, 150.
RSUN = 8.1
XR, ZR = 20., 10.
NB = 140           # background binning
# The born-cold layer is binned coarser than the background: it is 3,300 stars
# before any R cut and 1,039 after R > 5, and at the background's resolution
# almost every occupied bin holds one star, so the map carries no contrast.
NB_FG = 56
cCOLD, cHOT = '#1F6FB2', '#FF6347'

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 14, 'axes.labelsize': 16,
    'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 13,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 140, 'savefig.dpi': 200,
})

# ------------------------------ the selection --------------------------------
k = np.load(C.OUT_DIR + '/merger_birth_vs_z0_kinematics.npz')
cat = np.load(C.OUT_DIR + '/z0_insitu_catalog.npz')
o = np.argsort(cat['ids']); ci = cat['ids'][o]
p = np.searchsorted(ci, k['ids'])
ok = (p < len(ci)) & (ci[np.minimum(p, len(ci) - 1)] == k['ids'])
ix = o[p[ok]]
ids = k['ids'][ok]
eos = (np.abs(k['z0_vphi'][ok]) < VPHI_MAX) & (cat['ecc'][ix] > ECC_MIN)
cold = eos & (k['birth_vphi'][ok] >= VPHI_SPLIT)
hot = eos & (k['birth_vphi'][ok] < VPHI_SPLIT)
print(f'merger-window sample {len(ids):,}; Eos-like {eos.sum():,}; '
      f'born-cold {cold.sum():,}, born-hot {hot.sum():,}')

# -------------------------- z = 0 positions ----------------------------------
s = snap_mod.load_snapshot(127, 4, snappath=C.SIM_DIR, verbose=False,
    loadlist=['ParticleIDs', 'Coordinates', 'Velocities', 'Masses',
              'GFM_StellarFormationTime'])
real = s.data['GFM_StellarFormationTime'] > 0
for key in list(s.data):
    s.data[key] = s.data[key][real]
sf = sub_mod.subfind(127, directory=C.SIM_DIR,
                     loadlist=['GroupFirstSub', 'SubhaloPos'])
cen = sf.data['SubhaloPos'][int(sf.data['GroupFirstSub'][0])]
util.CentreOnHalo(s, cen)
util.align_galaxy(s, radialcut=.01)
c = s.data['Coordinates'] * 1e3
# align_galaxy puts the disc axis on component 0; cyclic map, determinant +1
POS = np.column_stack([c[:, 1], c[:, 2], c[:, 0]])
SID = s.data['ParticleIDs']
del s, c; gc.collect()

os_ = np.argsort(SID); ss = SID[os_]
pp = np.searchsorted(ss, ids)
hit = (pp < len(ss)) & (ss[np.minimum(pp, len(ss) - 1)] == ids)
P = np.full((len(ids), 3), np.nan)
P[hit] = POS[os_[pp[hit]]]
del POS, SID; gc.collect()
fin = np.isfinite(P[:, 0])
print(f'matched to a z = 0 position: {hit.sum():,} of {len(ids):,}')

R = np.hypot(P[:, 0], P[:, 1])
if R_MIN > 0:
    keep = R > R_MIN
    print(f'cylindrical cut R > {R_MIN:g} kpc: keeps {keep.sum():,} of {len(R):,} '
          f'({100 * keep.mean():.1f} %); born-cold {int((cold & keep).sum()):,} '
          f'of {int(cold.sum()):,} ({100 * (cold & keep).sum() / cold.sum():.1f} %)')
    fin = fin & keep
for lab, m in (('all merger-window', fin), ('Eos-like', eos & fin),
               ('born-cold', cold & fin), ('born-hot', hot & fin)):
    print(f'  {lab:<18s} N = {m.sum():>7,}  median R = {np.median(R[m]):5.2f} kpc, '
          f'median |z| = {np.median(np.abs(P[m, 2])):5.2f} kpc, '
          f'|z| 84th = {np.percentile(np.abs(P[m, 2]), 84):5.2f}, '
          f'{100 * (np.abs(R[m] - RSUN) < 2).mean():4.1f} % within 2 kpc of R_sun')

# ---------------------------------- figure -----------------------------------
fig, AX = plt.subplots(1, 2, figsize=(15.0, 5.6),
                       gridspec_kw=dict(width_ratios=[1., 1.], wspace=.34))
th = np.linspace(0, 2 * np.pi, 300)
VIEWS = [(0, 1, '$y$ [kpc]', XR, '(a)  face-on'),
         (0, 2, '$z$ [kpc]', ZR, '(b)  edge-on')]
for ax, (i, j, ylab, yr, tag) in zip(AX, VIEWS):
    bg = fin
    ax.hist2d(P[bg, i], P[bg, j], bins=(NB, int(NB * yr / XR)),
              range=[[-XR, XR], [-yr, yr]], cmap='Greys',
              norm=LogNorm(vmin=1, vmax=None), alpha=.55, rasterized=True)
    m = cold & fin
    h, xe, ye = np.histogram2d(P[m, i], P[m, j],
                               bins=(NB_FG, max(int(NB_FG * yr / XR), 8)),
                               range=[[-XR, XR], [-yr, yr]])
    pcm = ax.pcolormesh(xe, ye, np.where(h > 0, h, np.nan).T, cmap='Blues',
                        norm=LogNorm(vmin=1, vmax=np.nanmax(h)), rasterized=True)
    cb = fig.colorbar(pcm, ax=ax, pad=.015, fraction=.046)
    cb.set_label('born-cold stars per bin', fontsize=13)
    cb.ax.tick_params(labelsize=12)
    if j == 1:
        ax.plot(RSUN * np.cos(th), RSUN * np.sin(th), color='#E8112D', lw=1.8,
                ls='--', label=f'$R_\\odot$ = {RSUN} kpc')
    else:
        for sgn in (+1, -1):
            ax.axvline(sgn * RSUN, color='#E8112D', lw=1.8, ls='--',
                       label=f'$R_\\odot$ = {RSUN} kpc' if sgn > 0 else None)
    ax.legend(loc='upper right', handlelength=1.6)
    ax.set(aspect='equal', xlim=(-XR, XR), ylim=(-yr, yr), xlabel='$x$ [kpc]',
           ylabel=ylab)
    bb = dict(fc='white', ec='none', alpha=.85, pad=2)
    ax.text(.025, .96, tag, transform=ax.transAxes, va='top', fontsize=15,
            fontweight='bold', bbox=bb)
AX[0].text(.025, .055, f'born-cold Eos-like, N = {(cold & fin).sum():,}\n'
           'grey: all stars with $t_{\\rm form}$ = 4.99$-$6.54 Gyr\n'
           + (f'cuts: $3 < r < 30$ kpc (in the data) and $R > {R_MIN:g}$ kpc'
              if R_MIN > 0 else 'cut: $3 < r < 30$ kpc, already in the data'),
           transform=AX[0].transAxes, va='bottom', fontsize=12.5,
           bbox=dict(fc='white', ec='none', alpha=.85, pad=2))
f = f'{OUT}/au18_eos_cold_z0_positions{TAG}.png'
fig.savefig(f, bbox_inches='tight')
print(f'\nsaved {f}')
