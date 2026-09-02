"""Publication figure: the Eos selection today, and what those stars were at birth.

Three panels, from the merger-window sample merger_birth_vs_z0_kinematics.npz
(t_form = 4.99-6.54 Gyr):

  (a) all merger-born stars at z = 0, with the Eos band drawn on
  (b) the stars passing the Eos cut, |v_phi| < 80 km/s and ecc > 0.6, at z = 0
  (c) the same 7,583 stars at birth, with the v_phi,birth = 150 km/s split

The argument runs left to right.  At z = 0 the Eos cut takes a horizontal slice
out of one rotating distribution.  At birth those same stars are two groups -- a
concentration still on the disc ridge near +220 km/s and a broader, slower one --
and the 150 km/s split falls in the dip between them.

Each panel is normalised to its own peak, so colour compares shape and not
abundance.  The colour map is truncated at the pale end so that sparsely
populated bins are still visible against white, and the kinematic cuts are drawn
in a warm colour that the cool map cannot swallow.  The Eos band and the birth
split never share a panel, so they use the same cut colour.

Writes Fig_paper/au18_vr_vphi_three.pdf and .png.
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
import config_au18 as C

OUT = '/data/hz420-2/EoS_explore/Fig_paper'
os.makedirs(OUT, exist_ok=True)
VPHI_MAX, ECC_MIN, VPHI_SPLIT = 80., 0.6, 150.
RNG = [[-400, 400], [-300, 400]]
# Square bins: 800/120 = 700/105 = 6.67 km/s on both axes, so that with
# aspect='equal' the pixels come out square rather than stretched.
NX, NY = 120, 105
CUT = '#E8112D'                       # the kinematic cuts, warm against a cool map

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
CMAP = ListedColormap(plt.get_cmap('YlGnBu')(np.linspace(.18, 1., 256)))

k = np.load(C.OUT_DIR + '/merger_birth_vs_z0_kinematics.npz')
cat = np.load(C.OUT_DIR + '/z0_insitu_catalog.npz')
o = np.argsort(cat['ids']); sid = cat['ids'][o]
p = np.searchsorted(sid, k['ids'])
ok = (p < len(sid)) & (sid[np.minimum(p, len(sid) - 1)] == k['ids'])
ix = o[p[ok]]
bvR, bvphi = k['birth_vR'][ok], k['birth_vphi'][ok]
zvR, zvphi = k['z0_vR'][ok], k['z0_vphi'][ok]
tform = cat['tform'][ix]
eos = (np.abs(zvphi) < VPHI_MAX) & (cat['ecc'][ix] > ECC_MIN)
hot, cold = eos & (bvphi < VPHI_SPLIT), eos & (bvphi >= VPHI_SPLIT)
T0, T1 = tform.min(), tform.max()
TMED = np.median(tform[eos])
print(f'merger-born {eos.size:,}; Eos-like {eos.sum():,}; '
      f'born hot {hot.sum():,}; born cold {cold.sum():,}')
print(f'window {T0:.2f}-{T1:.2f} Gyr; Eos median t_form {TMED:.2f} Gyr '
      f'(all merger-born: {np.median(tform):.2f})')

# The window is quoted rounded, 5.0-6.5 Gyr; the sample itself spans
# 4.99-6.54 Gyr (T0, T1 above), which is what the selection actually used.
# The right-hand panel carries no epoch: each star is measured at its own birth
# snapshot, spread over that whole window.
TITLES = [r'All stars with $5.0 < t_{\rm form} < 6.5$ Gyr, at $z=0$',
          r'Selected Eos-like stars, at $z=0$',
          r'Selected Eos-like stars, at birth']

fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.1), sharex=True, sharey=True)
panels = [(zvR, zvphi, np.ones(len(zvR), bool), True, False, '(a)'),
          (zvR, zvphi, eos, True, False, '(b)'),
          (bvR, bvphi, eos, False, True, '(c)')]
for ax, (x, y, m, band, split, tag), title in zip(axes, panels, TITLES):
    h, xe, ye = np.histogram2d(x[m], y[m], bins=[NX, NY], range=RNG)
    h = np.where(h > 0, h / h.max(), np.nan)
    im = ax.pcolormesh(xe, ye, h.T, cmap=CMAP, norm=LogNorm(vmin=1e-3, vmax=1),
                       rasterized=True)
    ax.axhline(0, color='.6', lw=.6, zorder=1)
    ax.axvline(0, color='.6', lw=.6, zorder=1)
    if band:
        ax.axhspan(-VPHI_MAX, VPHI_MAX, color=CUT, alpha=.07, lw=0, zorder=2)
        for v in (-VPHI_MAX, VPHI_MAX):
            ax.axhline(v, color=CUT, lw=2.0, ls='--', zorder=3)
    if split:
        ax.axhline(VPHI_SPLIT, color=CUT, lw=2.4, ls='--', zorder=3)
        bb = dict(fc='white', ec='none', alpha=.88, pad=3.0)
        ax.annotate('born-cold', (385, 350), fontsize=20, ha='right',
                    va='center', color=CUT, bbox=bb, zorder=4)
        # both labels in the sparse right-hand corners, clear of the two lobes
        # (hyphenated, matching the born-cold / born-hot usage in the text)
        ax.annotate('born-hot', (385, -230), fontsize=20, ha='right',
                    va='center', color=CUT, bbox=bb, zorder=4)
    ax.text(.035, .955, tag, transform=ax.transAxes, va='top', fontsize=16,
            fontweight='bold')
    ax.set_title(title, fontsize=13.5, pad=8)
    ax.set_xlabel(r'$v_R$ [km s$^{-1}$]')
    # equal scale on both axes: 1 km/s is the same length in x and y
    ax.set(aspect='equal', xlim=(-400, 400), ylim=(-300, 400),
           xticks=np.arange(-400, 401, 200), yticks=np.arange(-300, 401, 100))
axes[0].set_ylabel(r'$v_\phi$ [km s$^{-1}$]')
cb = fig.colorbar(im, ax=axes, fraction=.020, pad=.012)
cb.set_label("density, normalised to each panel's peak")
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/au18_vr_vphi_three.{ext}', bbox_inches='tight')
print(f'saved {OUT}/au18_vr_vphi_three.pdf and .png')
