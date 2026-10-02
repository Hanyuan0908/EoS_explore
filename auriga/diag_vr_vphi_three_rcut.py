"""Diagnostic, NOT a paper figure: au18_vr_vphi_three with a radial cut.

The three panels of fig_paper_vr_vphi_three -- all merger-window stars at z = 0,
the Eos selection, and those same stars at birth -- with an extra cylindrical cut
R > R_MIN kpc applied at z = 0, to step outside the bar.

    usage:  diag_vr_vphi_three_rcut.py [R_MIN]        (default 5)

WHY ONE MIGHT WANT THIS.  At z = 0 the born-cold half of the Eos selection is a
bar: face-on it is elongated and bilobed, and 68.5 per cent of it sits inside
R = 5 kpc (diag_eos_cold_z0_positions).  Anything read off the paper figure about
the born-cold channel is therefore read mostly off the bar.

WHAT IS ALREADY IN THE DATA, and is not optional.  The parent file
merger_birth_vs_z0_kinematics.npz carries a spherical aperture 3 < r < 30 kpc:
the minimum kept radius is exactly 3.000 kpc, 0 per cent of stars at
r = 2.5-3.0 survive, and the cut removes 164,436 of the 336,262 in-situ stars
formed in this window -- 49 per cent, overwhelmingly central (median r = 1.37 kpc
against 7.35 kpc for those kept).  The paper figure's panel (a) is therefore not
"all stars with 5.0 < t_form < 6.5 Gyr"; it is the 51 per cent of them outside
3 kpc.  R_MIN only moves that inner edge further out.

Panels stay normalised to their own peak, as in the paper version, so colour
shows shape and not abundance; the counts are printed and annotated instead.

Writes figures/au18_vr_vphi_three_R<N>.png -- figures/, not Fig_paper/.
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
import config_au18 as C

import sys
OUT = C.FIG_DIR
os.makedirs(OUT, exist_ok=True)
R_MIN = float(sys.argv[1]) if len(sys.argv) > 1 else 5.
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
# the extra cut, applied to EVERY panel including the parent in (a)
Rz0 = cat['R'][ix]
keep = Rz0 > R_MIN
eos = (np.abs(zvphi) < VPHI_MAX) & (cat['ecc'][ix] > ECC_MIN) & keep
hot, cold = eos & (bvphi < VPHI_SPLIT), eos & (bvphi >= VPHI_SPLIT)
n0 = int((np.abs(zvphi) < VPHI_MAX).sum() & 0) or None
print(f'cylindrical cut R > {R_MIN:g} kpc at z = 0: keeps {keep.sum():,} of '
      f'{keep.size:,} ({100 * keep.mean():.1f} %)')
eos_nocut = (np.abs(zvphi) < VPHI_MAX) & (cat['ecc'][ix] > ECC_MIN)
print(f'  Eos-like      {eos.sum():,} of {eos_nocut.sum():,} survive '
      f'({100 * eos.sum() / eos_nocut.sum():.1f} %)')
for lab, a_, b_ in (('born-hot ', hot, eos_nocut & (bvphi < VPHI_SPLIT)),
                    ('born-cold', cold, eos_nocut & (bvphi >= VPHI_SPLIT))):
    print(f'  {lab}     {a_.sum():,} of {b_.sum():,} survive '
          f'({100 * a_.sum() / b_.sum():.1f} %)')
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
TITLES = [t + f'\n($R > {R_MIN:g}$ kpc)' for t in TITLES]

fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.1), sharex=True, sharey=True)
panels = [(zvR, zvphi, keep, True, False, '(a)'),
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
    ax.text(.035, .055, f'N = {int(m.sum()):,}', transform=ax.transAxes,
            va='bottom', fontsize=13,
            bbox=dict(fc='white', ec='none', alpha=.85, pad=2))
    ax.set_title(title, fontsize=13.5, pad=8)
    ax.set_xlabel(r'$v_R$ [km s$^{-1}$]')
    # equal scale on both axes: 1 km/s is the same length in x and y
    ax.set(aspect='equal', xlim=(-400, 400), ylim=(-300, 400),
           xticks=np.arange(-400, 401, 200), yticks=np.arange(-300, 401, 100))
axes[0].set_ylabel(r'$v_\phi$ [km s$^{-1}$]')
cb = fig.colorbar(im, ax=axes, fraction=.020, pad=.012)
cb.set_label("density, normalised to each panel's peak")
f = f'{OUT}/au18_vr_vphi_three_R{R_MIN:g}.png'
fig.savefig(f, bbox_inches='tight')
print(f'saved {f}')
