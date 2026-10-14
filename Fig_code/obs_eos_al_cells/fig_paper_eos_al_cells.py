"""Publication figure (observational): orbits across the [Al/Fe]-[Fe/H] plane and inside the Eos cloud.

Two panels, (a) r_apo and (b) z_max:
  background  APOGEE halo-orbit stars (ecc>0.7 or Lz<0), pixels 0.05 x 0.03 dex (>=8 stars) coloured by the mean;
  cells       4 [Al/Fe] bands from the -0.12 selection floor to the 95th pct of Eos [Al/Fe], sides of slope 1 at the
              5th/95th pct of Eos in w=[Fe/H]-[Al/Fe] (hand-drawn shape); filled with the mean of the EOS stars inside;
  points      canonical Eos stars (n=353), each coloured by its own value;
  lines       GS/E (reference accreted mask) and high-alpha in-situ running medians of [Al/Fe].
Cell design and statistics: scripts_repro/eos_al_axis_cells.py (308 of the 353 Eos stars fall in the cells).
Writes Fig_paper/obs_eos_al_cells.pdf and .png.
"""
import os
import sys
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
from scipy.stats import binned_statistic_2d

REPO = '/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore'
sys.path.insert(0, REPO + '/scripts_repro')
import eos_tests_common as E                                     # noqa: E402
from al_levels_orbits import lines                               # noqa: E402
from eos_al_axis_cells import design, stats                      # noqa: E402

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 16, 'axes.labelsize': 20, 'axes.titlesize': 18,
    'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 14,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})

OUT = REPO + '/Fig_paper'
FEH_R, AL_R = (-1.6, 0.0), (-0.45, 0.45)
NBX, NBY = 32, 30
ALPHA_BG, ALPHA_FG = 0.55, 0.7
PANELS = [('rap', E.rap, r'$r_{\rm apo}$ [kpc]', (4, 20)),
          ('zmax', E.zmax, r'$z_{\max}$ [kpc]', (1, 10))]
CELL_LAB = {'rap': r'$\langle r_{\rm apo}\rangle$', 'zmax': r'$\langle z_{\max}\rangle$'}


def tag(a, t):
    a.text(0.035, 0.965, t, transform=a.transAxes, fontsize=17, fontweight='bold', va='top', ha='left')


def main():
    D = design()
    rows = stats(D)
    al_gse, al_high, (gc, gm, hc, hm, p, g_end) = lines()
    halo = E.halo & np.isfinite(E.al) & np.isfinite(E.feh) & np.isfinite(E.rap) & np.isfinite(E.zmax)
    xe = np.linspace(*FEH_R, NBX + 1); ye = np.linspace(*AL_R, NBY + 1)
    ff = np.linspace(*FEH_R, 200)
    cmap = plt.get_cmap('RdYlBu_r')

    fig, axes = plt.subplots(1, 2, figsize=(19.5, 6.6), sharey=True, gridspec_kw=dict(wspace=0.18))
    for ax, (key, q, lab, vlim), t in zip(axes, PANELS, ('(a)', '(b)')):
        norm = Normalize(*vlim)
        mean = binned_statistic_2d(E.feh[halo], E.al[halo], q[halo], 'mean', bins=[xe, ye]).statistic
        cnt = binned_statistic_2d(E.feh[halo], E.al[halo], None, 'count', bins=[xe, ye]).statistic
        img = np.where(cnt >= 8, mean, np.nan)
        ax.imshow(img.T, origin='lower', extent=[*FEH_R, *AL_R], aspect='auto', cmap=cmap, norm=norm,
                  alpha=ALPHA_BG, zorder=0, rasterized=True)
        for poly, r in zip(D['polys'], rows):
            ax.add_patch(Polygon(poly, closed=True, fc=cmap(norm(r[key])), ec='none', alpha=ALPHA_FG, zorder=2))
        e = E.eos
        ax.scatter(E.feh[e], E.al[e], s=20, c=q[e], cmap=cmap, norm=norm, edgecolors='k', linewidths=0.35,
                   zorder=3, rasterized=True)
        for poly in D['polys']:
            ax.add_patch(Polygon(poly, closed=True, fill=False, ec='k', lw=1.4, zorder=4))
        for poly, r in zip(D['polys'], rows):
            ax.text(0.5 * (poly[0, 0] + poly[3, 0]) - 0.03, poly[:, 1].mean(), CELL_LAB[key] + f' = {r[key]:.1f}', va='center', ha='right',
                    fontsize=14, zorder=6, bbox=dict(fc='white', ec='none', alpha=0.8, pad=1))
        ax.plot(ff[ff <= g_end], al_gse(ff[ff <= g_end]), color='k', lw=2.2, zorder=5, label='GS/E')
        ax.plot(ff, al_high(ff), color='seagreen', lw=2.2, zorder=5, label=r'high-$\alpha$')
        ax.axhline(E.ALFE_CUT, color='0.3', ls=':', lw=1.3, zorder=1)
        ax.scatter([], [], s=30, c='0.6', edgecolors='k', linewidths=0.35, label='Eos')
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.05)
        cb.set_label(lab, fontsize=25); cb.ax.tick_params(labelsize=16)
        ax.set_xlim(*FEH_R); ax.set_ylim(*AL_R)
        ax.set_xlabel('[Fe/H]', fontsize=25)
        if ax is axes[0]:
            ax.set_ylabel('[Al/Fe]', fontsize=25)
        ax.legend(loc='lower right', handlelength=1.6, fontsize=19, markerscale=1.6)
        tag(ax, t)
    os.makedirs(OUT, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/obs_eos_al_cells.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('cells:', [(r['n'], round(r['rap'], 1), round(r['zmax'], 1)) for r in rows])
    print('wrote', f'{OUT}/obs_eos_al_cells.pdf/.png')


if __name__ == '__main__':
    main()
