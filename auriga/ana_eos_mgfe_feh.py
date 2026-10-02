"""The two mock-Eos channels of Au18 in the [Mg/Fe]-[Fe/H] plane.

Same sample as au18_rbirth_feh_jr: merger-window stars (t_form = 4.99-6.54 Gyr)
passing the Eos cut at z = 0, split by birth kinematics into born-cold
(v_phi,birth >= 150 km/s) and born-hot.  Abundances come from
out/z0_insitu_catalog.npz (GFM_Metals through config_au18.bracket_abundance).

READ THIS BEFORE USING THE FIGURE.  Auriga's [Mg/Fe] is not the Milky Way's:

  * the zero point is offset -- the whole galaxy sits near [Mg/Fe] = -0.30,
    not around 0.0, because Auriga's yields do not reproduce the solar Mg/Fe
    ratio;
  * the dynamic range is ~0.04-0.07 dex (16-84 per cent), against the ~0.3 dex
    that separates the MW's high- and low-alpha sequences.  There is no
    alpha-bimodality in this model to find;
  * the two populations differ by 0.006 dex in median [Mg/Fe] -- a thousandth of
    the spread of the MW sequences, i.e. chemically indistinguishable in alpha.

Every other alpha element behaves the same way: O, Si and Ne all have 16-84
spreads of 0.03-0.05 dex and separate the two populations by <= 0.01 dex.  So the
separation between these channels in Au18 is in [Fe/H] (-0.16 vs -0.40) and in
kinematics, not in alpha, and this plane cannot be used the way an observer uses
it.  The y axis is scaled to the data rather than to a Milky-Way-like range, so
what little structure exists is visible; do not read the spread as physical
without that caveat.

Writes figures/au18_eos_mgfe_feh.png.
"""
import os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import config_au18 as C
import eos_origins as EO
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import orbit_tools as OT

cCOLD, cHOT = '#1F6FB2', '#FF6347'
FRNG, MRNG = (-1.0, 0.5), (-0.42, -0.16)

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 13, 'axes.labelsize': 14.5,
    'xtick.labelsize': 12.5, 'ytick.labelsize': 12.5, 'legend.fontsize': 11,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'figure.dpi': 140, 'savefig.dpi': 200,
})

d = EO.load()
cat = np.load(C.OUT_DIR + '/z0_insitu_catalog.npz')
o = np.argsort(cat['ids']); cid = cat['ids'][o]
p = np.searchsorted(cid, d['ids'])
ok = (p < len(cid)) & (cid[np.minimum(p, len(cid) - 1)] == d['ids'])
idx = o[p[ok]]
mg = np.full(len(d['ids']), np.nan); mg[ok] = cat['mgfe'][idx]
fe = d['feh']
POPS = [('born-cold', d['disc_born'], cCOLD), ('born-hot', d['halo_born'], cHOT)]

fig, ax = plt.subplots(figsize=(8.2, 6.0))
fin = np.isfinite(fe) & np.isfinite(mg)
h, xe, ye = np.histogram2d(fe[fin], mg[fin], bins=(120, 90), range=[FRNG, MRNG])
pcm = ax.pcolormesh(xe, ye, np.where(h > 0, h, np.nan).T, cmap='Greys',
                    norm=LogNorm(vmin=1, vmax=h.max()), rasterized=True)
cb = fig.colorbar(pcm, ax=ax, pad=.015, fraction=.045)
cb.set_label('stars per bin, all born in the window', fontsize=11.5)

for lab, m, c in POPS:
    OT.density_contours(ax, fe[m & fin], mg[m & fin], [list(FRNG), list(MRNG)], c,
                        levels=(0.9, 0.6, 0.3), bins=60, smooth=1.5, lw=2.0,
                        label=f'{lab} ({(m & fin).sum():,})')
    ax.plot(np.nanmedian(fe[m & fin]), np.nanmedian(mg[m & fin]), '*', color=c,
            ms=17, mec='w', mew=.9, zorder=5)

ax.text(.985, .04,
        'medians differ by 0.006 dex in [Mg/Fe]\n'
        'Au18 spans ~0.05 dex in total, against ~0.3 dex\n'
        'between the Milky Way sequences: no alpha-bimodality here',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=10, color='.25',
        bbox=dict(fc='white', ec='.8', alpha=.9, pad=3))
ax.set(xlim=FRNG, ylim=MRNG, xlabel='[Fe/H]', ylabel='[Mg/Fe]')
ax.set_title('Au18 mock Eos in the chemical plane (note the compressed y axis)',
             fontsize=12.5)
ax.legend(loc='upper right', handlelength=1.8)

fig.tight_layout()
os.makedirs(C.FIG_DIR, exist_ok=True)
out = C.FIG_DIR + '/au18_eos_mgfe_feh.png'
fig.savefig(out, bbox_inches='tight')

# ------------------------------------------------------------------- numbers --
for lab, m, _ in POPS:
    q = lambda v: np.nanpercentile(v[m & fin], [16, 50, 84])
    a, b = q(mg), q(fe)
    print(f'{lab:10s} ({(m & fin).sum():,})  [Mg/Fe] {a[1]:+.3f} '
          f'(16-84 {a[2] - a[0]:.3f})   [Fe/H] {b[1]:+.3f} (16-84 {b[2] - b[0]:.3f})')
cold, hot = d['disc_born'] & fin, d['halo_born'] & fin
print(f'difference hot - cold: [Mg/Fe] {np.nanmedian(mg[hot]) - np.nanmedian(mg[cold]):+.3f} dex, '
      f'[Fe/H] {np.nanmedian(fe[hot]) - np.nanmedian(fe[cold]):+.3f} dex')
print('\nsaved', out)
