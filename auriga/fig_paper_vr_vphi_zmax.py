"""Publication figure: the Eos selection with a z_max cut, and what those stars
were at birth.

Nothing is annotated on the panels: the counts and the z_max condition belong in
the caption, not on the figure.  The script prints every number it would have
written.  The colourbar is placed from the drawn position of the last axes rather
than with ax=axes, because aspect='equal' leaves each axes shorter than its
gridspec slot and a slot-spanning bar overshoots the panels top and bottom.

The three panels of au18_vr_vphi_three, with two changes.

1. THE Eos CUT GAINS z_max > 2 kpc.  |v_phi| < 80 km/s and ecc > 0.6 alone admit
   a large population of bar stars: at z = 0 the born-cold half of the old
   selection is visibly barred, elongated and bilobed, with 68.5 per cent of it
   inside R = 5 kpc.  z_max is the right quantity to cut on rather than R or
   instantaneous |z|: R keeps bar stars currently at large radius, |z| is a
   snapshot of orbital phase and keeps a bar star caught at the top of its small
   vertical oscillation while discarding a halo star crossing the plane, and
   z_max is a property of the orbit.  It is applied ONLY to the selection; panel
   (a) is every star of the epoch, uncut.

2. THE PARENT IS THE WHOLE IN-SITU SAMPLE.  au18_vr_vphi_three reads
   merger_birth_vs_z0_kinematics.npz, which carries an undocumented spherical
   aperture 3 < r < 30 kpc: the minimum kept radius is exactly 3.000 kpc and the
   aperture removes 164,436 of the 336,262 in-situ stars formed in that window --
   49 per cent, overwhelmingly central (median r = 1.37 kpc).  Panel (a) there is
   therefore not "all stars" as its title claims.  Here birth velocities are
   remeasured from the snapshots (diag_vr_vphi_window) so no aperture is needed,
   and the window is 5.0-6.5 Gyr exactly, which is what the title has always
   said; the old sample actually spanned 4.99-6.54.

   COUNTS WILL NOT MATCH CONVENTIONS.md, which quotes numbers inheriting the
   aperture (171,826 / 7,583 / 4,283 / 3,300).

z_max comes from integrating each orbit for 30 circular periods in the
snapshot-127 AGAMA CylSpline (prep_zmax_z0.py).  That potential is AXISYMMETRIC,
so the bar is averaged out of the integration -- which is what makes the cut work
as bar removal, but means these are not the true orbits of bar stars.  A z_max
cut is also not selection-neutral: it correlates with being dynamically hot and
will raise the born-hot fraction partly for reasons unconnected with the bar.
Both belong in the caption.

Birth velocities are taken from any cached window that encloses this one, since
the measurement is per star; prep_zmax_z0.py 4.5 6.5 supplies z_max the same way.
Run diag_vr_vphi_window.py for a window no cache covers.

Writes Fig_paper/au18_vr_vphi_zmax.pdf and .png.
"""
import os, re, sys
import glob
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
import config_au18 as C

OUT = '/data/hz420-2/EoS_explore/Fig_paper'
os.makedirs(OUT, exist_ok=True)
T_LO, T_HI = 5.0, 6.0
VPHI_MAX, ECC_MIN, VPHI_SPLIT, ZMAX_MIN = 80., 0.6, 150., 2.
RNG = [[-400, 400], [-300, 400]]
# Square bins: 800/120 = 700/105 = 6.67 km/s both ways, so with aspect='equal'
# the pixels are square.
NX, NY = 120, 105
CUTC = '#E8112D'

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    # house sizes x 1.15, matching the other recent paper figures
    'font.size': 15.5, 'axes.labelsize': 17,
    'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 14.5,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})
CMAP = ListedColormap(plt.get_cmap('YlGnBu')(np.linspace(.18, 1., 256)))

cat = np.load(C.OUT_DIR + '/z0_insitu_catalog.npz')
win = (cat['tform'] >= T_LO) & (cat['tform'] < T_HI)
ids = cat['ids'][win]

# Birth velocities come from the cache for this window, or from any cached
# window that ENCLOSES it -- the measurement is per star, so a sub-window is just
# a subset and need not be re-integrated.
bc = None
for f in sorted(glob.glob(C.OUT_DIR + '/birth_vrvphi_*.npz')):
    m = re.match(r'.*birth_vrvphi_([\d.]+)_([\d.]+)\.npz$', f)
    if m and float(m.group(1)) <= T_LO and float(m.group(2)) >= T_HI:
        bc = f
        break
if bc is None:
    raise SystemExit(f'no cached birth velocities covering {T_LO}-{T_HI} Gyr; '
                     f'run "diag_vr_vphi_window.py {T_LO} {T_HI}" first')
b = np.load(bc)
ob = np.argsort(b['ids']); bi = b['ids'][ob]
pb = np.searchsorted(bi, ids)
hb = (pb < len(bi)) & (bi[np.minimum(pb, len(bi) - 1)] == ids)
bvR = np.full(len(ids), np.nan); bvphi = np.full(len(ids), np.nan)
bvR[hb] = b['bvR'][ob[pb[hb]]]
bvphi[hb] = b['bvphi'][ob[pb[hb]]]
print(f'birth velocities from {os.path.basename(bc)}: '
      f'{hb.sum():,} of {len(ids):,} matched ({100 * hb.mean():.1f} %)')

zf = C.OUT_DIR + '/zmax_z0_4.5_6.5.npz'
if not os.path.exists(zf):
    raise SystemExit(f'missing {zf}; run "prep_zmax_z0.py 4.5 6.5" first')
zc = np.load(zf)
oz = np.argsort(zc['ids']); zi = zc['ids'][oz]
pz = np.searchsorted(zi, ids)
hz = (pz < len(zi)) & (zi[np.minimum(pz, len(zi) - 1)] == ids)
zmax = np.full(len(ids), np.nan)
zmax[hz] = zc['zmax'][oz[pz[hz]]]

zvR, zvphi, ecc = cat['vR'][win], cat['vphi'][win], cat['ecc'][win]
par = np.isfinite(bvR) & np.isfinite(bvphi)          # panel (a): everything
eos = par & (np.abs(zvphi) < VPHI_MAX) & (ecc > ECC_MIN) & (zmax > ZMAX_MIN)
hot, cold = eos & (bvphi < VPHI_SPLIT), eos & (bvphi >= VPHI_SPLIT)
eos_noz = par & (np.abs(zvphi) < VPHI_MAX) & (ecc > ECC_MIN)
print(f'window {T_LO}-{T_HI} Gyr, in-situ, no aperture: {par.sum():,}')
print(f'  z_max matched for {hz.sum():,} ({100 * hz.mean():.1f} %)')
print(f'  |v_phi| < {VPHI_MAX:.0f} and ecc > {ECC_MIN}        : {eos_noz.sum():,}')
print(f'  ... and z_max > {ZMAX_MIN:g} kpc (the Eos cut) : {eos.sum():,} '
      f'({100 * eos.sum() / eos_noz.sum():.1f} % of them, '
      f'{100 * eos.sum() / par.sum():.2f} % of the parent)')
print(f'    born-hot  {hot.sum():,}  ({100 * hot.sum() / eos.sum():.1f} %)')
print(f'    born-cold {cold.sum():,}  ({100 * cold.sum() / eos.sum():.1f} %)')

TIT = [rf'All stars with ${T_LO} < t_{{\rm form}} < {T_HI}$ Gyr, at $z=0$',
       r'Selected Eos-like stars, at $z=0$',
       r'Selected Eos-like stars, at birth']
fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.4), sharex=True, sharey=True)
panels = [(zvR, zvphi, par, True, False, '(a)'),
          (zvR, zvphi, eos, True, False, '(b)'),
          (bvR, bvphi, eos, False, True, '(c)')]
for ax, (x, y, m, band, split, tag), title in zip(axes, panels, TIT):
    h, xe, ye = np.histogram2d(x[m], y[m], bins=[NX, NY], range=RNG)
    h = np.where(h > 0, h / h.max(), np.nan)
    im = ax.pcolormesh(xe, ye, h.T, cmap=CMAP, norm=LogNorm(vmin=1e-3, vmax=1),
                       rasterized=True)
    ax.axhline(0, color='.6', lw=.6, zorder=1)
    ax.axvline(0, color='.6', lw=.6, zorder=1)
    if band:
        ax.axhspan(-VPHI_MAX, VPHI_MAX, color=CUTC, alpha=.07, lw=0, zorder=2)
        for v in (-VPHI_MAX, VPHI_MAX):
            ax.axhline(v, color=CUTC, lw=2.0, ls='--', zorder=3)
    if split:
        ax.axhline(VPHI_SPLIT, color=CUTC, lw=2.4, ls='--', zorder=3)
        bb = dict(fc='white', ec='none', alpha=.88, pad=3.0)
        ax.annotate('born-cold', (385, 350), fontsize=21, ha='right', va='center',
                    color=CUTC, bbox=bb, zorder=4)
        ax.annotate('born-hot', (385, -230), fontsize=21, ha='right', va='center',
                    color=CUTC, bbox=bb, zorder=4)
    ax.text(.035, .955, tag, transform=ax.transAxes, va='top', fontsize=18,
            fontweight='bold')
    ax.set_title(title, fontsize=14.5, pad=8)
    ax.set_xlabel(r'$v_R$ [km s$^{-1}$]')
    ax.set(aspect='equal', xlim=(-400, 400), ylim=(-300, 400),
           xticks=np.arange(-400, 401, 200), yticks=np.arange(-300, 401, 100))
axes[0].set_ylabel(r'$v_\phi$ [km s$^{-1}$]')
fig.canvas.draw()
pos = axes[-1].get_position()
cax = fig.add_axes([pos.x1 + .012, pos.y0, .011, pos.height])
cb = fig.colorbar(im, cax=cax)
cb.set_label('renormalised density')
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/au18_vr_vphi_zmax.{ext}', bbox_inches='tight')
print(f'\nsaved {OUT}/au18_vr_vphi_zmax.pdf and .png')
