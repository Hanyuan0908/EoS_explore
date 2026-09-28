"""Gaia-XP (Li+2024) [M/H]-[a/M] plane, two panels:
  (a) ALL good-quality stars (e_moh<0.1, e_aom<0.05, plx/e_plx>5) -- log density.
  (b) halo kinematic cut (ecc>0.7 | Lz<0) -- mean z_max pixel map,
      exactly as in plot_eos_branches_ab_zmaxcol.py (APOGEE).

Kinematics (ecc, r_apo, r_peri, z_max, Lz) from AGAMA + McMillan17, cached by
cache_xp_kinematics.py -> data_repro/xp_kinematics.npz. Axes are the XP [M/H] and
[a/M]; the APOGEE branch dividers / Eos segment are overplotted for reference (they
were defined on APOGEE [Fe/H]/[Mg/Fe], so treat the overlay as a guide).
Output figures_repro/01_xp_branches_ab.png.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

REPO = '/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore'
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 15, 'axes.labelsize': 18, 'axes.titlesize': 17,
    'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 13,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 130, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})

d = np.load(REPO + '/data_repro/xp_kinematics.npz')
feh = d['moh'].astype(float); mg = d['aom'].astype(float)
lz = d['lz'].astype(float); ecc = d['ecc'].astype(float); zmax = d['zmax'].astype(float)
halo = (ecc > 0.7) | (lz < 0)
print('quality N:', len(feh), '| halo N:', int(halo.sum()))

XR = (-2.0, 0.6); YMG = (-0.1, 0.42)
VMIN, VMAX = 1.0, 8.0


def acc(f):     return -0.3 * f - 0.1
def hl(f):      return -0.14 * f + 0.135
def divline(f): return 0.317 * f + 0.353


def mg_lines(ax):
    xx = np.linspace(*XR, 50)
    ax.plot(xx, acc(xx), color='k', ls='--', lw=1.7, zorder=3)
    ax.plot(xx, hl(xx),  color='k', ls=':',  lw=2.1, zorder=3)
    xe = np.linspace(-0.9, -0.2, 30)
    ax.plot(xe, divline(xe), color='lime', ls='-', lw=2.6, zorder=4)


def tag(a, t):
    a.text(0.03, 0.965, t, transform=a.transAxes, fontsize=18, fontweight='bold',
           va='top', ha='left', bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5))


fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15.5, 6.4), constrained_layout=True)

# (a) full-sample density
s = np.isfinite(feh) & np.isfinite(mg)
h, xe, ye = np.histogram2d(feh[s], mg[s], bins=[80, 60], range=[XR, YMG])
him = np.full_like(h, np.nan); him[h > 0] = np.log10(h[h > 0])
ax_a.imshow(him.T, origin='lower', extent=[*XR, *YMG], aspect='auto', cmap='Greys',
            vmin=np.nanpercentile(him, 2), vmax=np.nanpercentile(him, 99), zorder=0)
mg_lines(ax_a)
ax_a.text(-0.62, 0.155, 'Eos', color='k', fontsize=21, fontweight='bold', zorder=5)
tag(ax_a, '(a)')
ax_a.text(0.97, 0.04, f'all quality XP stars\n$n={int(s.sum()):,}$', transform=ax_a.transAxes,
          ha='right', va='bottom', fontsize=13)
ax_a.set_xlim(*XR); ax_a.set_ylim(*YMG)
ax_a.set_xlabel('[M/H]  (XP)'); ax_a.set_ylabel(r'[$\alpha$/M]  (XP)')

# (b) halo cut, mean zmax
sR = halo & np.isfinite(feh) & np.isfinite(mg)
med = binned_statistic_2d(feh[sR], mg[sR], zmax[sR], statistic='mean', bins=(50, 38), range=[XR, YMG]).statistic
cnt = binned_statistic_2d(feh[sR], mg[sR], None, statistic='count', bins=(50, 38), range=[XR, YMG]).statistic
med[cnt < 3] = np.nan
im_b = ax_b.imshow(med.T, origin='lower', extent=[*XR, *YMG], aspect='auto',
                   cmap='RdYlBu_r', vmin=VMIN, vmax=VMAX, zorder=0)
mg_lines(ax_b)
ax_b.text(-0.62, 0.155, 'Eos', color='k', fontsize=21, fontweight='bold', zorder=5)
tag(ax_b, '(b)')
ax_b.text(0.97, 0.04, f'halo: $e>0.7$ or $L_z<0$\n$n={int(sR.sum()):,}$', transform=ax_b.transAxes,
          ha='right', va='bottom', fontsize=13)
ax_b.set_xlim(*XR); ax_b.set_ylim(*YMG)
ax_b.set_xlabel('[M/H]  (XP)'); ax_b.set_ylabel(r'[$\alpha$/M]  (XP)')
cb = fig.colorbar(im_b, ax=ax_b, pad=0.01); cb.set_label(r'mean $z_{\rm max}$ [kpc]')

fig.savefig(REPO + '/figures_repro/01_xp_branches_ab.png', bbox_inches='tight')
print('wrote figures_repro/01_xp_branches_ab.png')
