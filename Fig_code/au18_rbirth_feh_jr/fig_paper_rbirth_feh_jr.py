"""Publication figure: where the two Eos populations were born, and their radial actions.

Layout is a corner plot plus one full-width panel underneath:

  (a) top    R_birth distributions of the two populations
  (b) main   R_birth against [Fe/H], greyscale = all stars born in the window
  (c) right  [Fe/H] distributions of the two populations
  (d) bottom radial action J_R

Panel (b) carries density contours for the two populations, enclosing 30, 60 and
90 per cent of each, over a greyscale of every star born in the window.  The
running medians of the earlier version are left off: the marginals on the top and
right make that comparison better, and three sets of lines over one map was too
much ink.

Panel (d) shows each population twice: the thick solid curve is the
solar-neighbourhood selection (within DMAX of a Sun at RSUN, pooled over NAZ
azimuths, since a single azimuth leaves only tens of stars), the thin dashed
curve is the whole population.  The gap between them is what the selection costs.

NAZ sets how many Sun positions are pooled.  NAZ = 1 is the literal single solar
neighbourhood and matches ana_eos_origins_solar.py, at the cost of leaving only
of order a hundred stars per population -- thin for a KDE, and the disc is not
axisymmetric, so one azimuth also samples whatever happens to lie in that
direction.  NAZ = 4 keeps the spheres disjoint (centres 11.5 kpc apart, against a
diameter of 8) and roughly triples the sample; NAZ = 8 puts the centres 6.2 kpc
apart so the spheres overlap and the union becomes a solar annulus instead.

Thin solid lines mark the median of each distribution.  They are solid rather
than dotted so that panel (d) carries only two line styles -- thick solid for the
solar selection, thin dashed for the whole population -- with thickness, not
style, separating a median from a curve.  In (d) they are drawn only
for the solar-selected curves, not for the dashed whole-population ones, which
would double the number of lines for little gain.  In (c) the marker is
horizontal because that panel's value axis is the vertical one.

Densities are Gaussian KDEs reflected at the hard bounds (R_birth > 0, J_R > 0)
with a robust bandwidth, min(sigma, IQR/1.349): J_R has a heavy tail and Scott's
rule, driven by that sigma, flattens the distribution into something that looks
nothing like the histogram.

Sample: the original merger window, t_form = 4.99-6.54 Gyr (eos_origins.py).
Writes Fig_paper/au18_rbirth_feh_jr.pdf and .png.
"""
import os, sys
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import config_au18 as C
import eos_origins as EO
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import orbit_tools as OT

OUT = '/data/hz420-2/EoS_explore/Fig_paper'
os.makedirs(OUT, exist_ok=True)
RSUN, DMAX, NAZ = 8.1, 4.0, 1
cCOLD, cHOT = '#1F6FB2', '#FF6347'
RRNG, FRNG, JRNG = (0., 22.), (-1.0, 0.5), (0., 3000.)

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

d = EO.load()
act = np.load(C.OUT_DIR + '/z0_actions.npz')
xy = np.load(C.OUT_DIR + '/z0_xy.npz')
o = np.argsort(act['ids']); aid = act['ids'][o]
p = np.searchsorted(aid, d['ids'])
ok = (p < len(aid)) & (aid[np.minimum(p, len(aid) - 1)] == d['ids'])
Jr = np.full(len(d['ids']), np.nan); Jr[ok] = act['Jr'][o[p[ok]]]
ox = np.argsort(xy['ids']); xid = xy['ids'][ox]
px = np.searchsorted(xid, d['ids'])
okx = (px < len(xid)) & (xid[np.minimum(px, len(xid) - 1)] == d['ids'])
X = np.full(len(d['ids']), np.nan); Y = np.full(X.shape, np.nan); Z = np.full(X.shape, np.nan)
ixx = ox[px[okx]]
X[okx], Y[okx], Z[okx] = xy['x'][ixx], xy['y'][ixx], xy['z'][ixx]

sun = np.zeros(len(X), bool)
for a in np.linspace(0, 2 * np.pi, NAZ, endpoint=False):
    sun |= np.sqrt((X - RSUN * np.cos(a)) ** 2 + (Y - RSUN * np.sin(a)) ** 2 + Z ** 2) < DMAX

COLD, HOT = d['disc_born'], d['halo_born']
POPS = [('born-cold', COLD, cCOLD), ('born-hot', HOT, cHOT)]
print(f'born-cold {COLD.sum():,}  born-hot {HOT.sum():,}')
print(f'solar selection (union of {NAZ} azimuths, R={RSUN}, d<{DMAX}): '
      f'born-cold {(COLD & sun).sum():,}  born-hot {(HOT & sun).sum():,}')


def kde(v, grid, lo=None, hi=None):
    """Reflected KDE with a robust bandwidth; see the module docstring."""
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 5 or np.std(v) == 0:
        return None
    sig = np.std(v); iqr = np.subtract(*np.percentile(v, [75, 25]))
    scale = min(sig, iqr / 1.349) if iqr > 0 else sig
    k = gaussian_kde(v, bw_method=0.9 * scale * len(v) ** (-0.2) / sig)
    out = k(grid)
    if lo is not None:
        out = out + k(2 * lo - grid); out[grid < lo] = 0.
    if hi is not None:
        out = out + k(2 * hi - grid); out[grid > hi] = 0.
    return out


# Nested gridspecs: the corner block (a, b, c) is drawn with almost no gap, while
# (d) needs a real gap beneath (b) so that (b) can carry its own x axis label.
fig = plt.figure(figsize=(9.6, 10.8))
outer = fig.add_gridspec(2, 1, height_ratios=[4.35, 2.5], hspace=.19,
                         left=.10, right=.975, top=.985, bottom=.062)
top = outer[0].subgridspec(2, 2, width_ratios=[4., 1.15],
                           height_ratios=[1.05, 3.3], hspace=.05, wspace=.05)
ax_t = fig.add_subplot(top[0, 0])
ax_m = fig.add_subplot(top[1, 0], sharex=ax_t)
ax_r = fig.add_subplot(top[1, 1], sharey=ax_m)
ax_b = fig.add_subplot(outer[1])

# --- (b) the main plane -------------------------------------------------------
fin = np.isfinite(d['R_birth']) & np.isfinite(d['feh'])
h, xe, ye = np.histogram2d(d['R_birth'][fin], d['feh'][fin], bins=(110, 90),
                           range=[RRNG, FRNG])
pcm = ax_m.pcolormesh(xe, ye, np.where(h > 0, h, np.nan).T, cmap='Greys',
                      norm=LogNorm(vmin=1, vmax=h.max()), rasterized=True)
cax = ax_m.inset_axes([.58, .085, .37, .028])
cb = fig.colorbar(pcm, cax=cax, orientation='horizontal')
cb.ax.tick_params(labelsize=10.5, length=2.5, pad=1.5)
cb.outline.set_linewidth(.7)
cax.set_title('stars per bin, all born in the window', fontsize=10.5, pad=3.5)
for lab, m, c in POPS:
    OT.density_contours(ax_m, d['R_birth'][m & fin], d['feh'][m & fin],
                        [list(RRNG), list(FRNG)], c, levels=(0.9, 0.6, 0.3),
                        bins=70, smooth=1.5, lw=2.0)
ax_m.set(xlim=RRNG, ylim=FRNG, xlabel=r'$R_{\rm birth}$ [kpc]', ylabel='[Fe/H]')
ax_m.text(.025, .05, '(b)', transform=ax_m.transAxes, fontsize=16, fontweight='bold')

# --- (a) R_birth marginal -----------------------------------------------------
gr = np.linspace(*RRNG, 400)
for lab, m, c in POPS:
    y = kde(d['R_birth'][m & fin], gr, lo=0.)
    ax_t.plot(gr, y, color=c, lw=2.4, label=f'{lab} ({m.sum():,})')
    ax_t.axvline(np.nanmedian(d['R_birth'][m & fin]), color=c, lw=1.1)
ax_t.set(ylim=(0, None), ylabel='density')
ax_t.tick_params(labelbottom=False)
ax_t.legend(loc='upper right', handlelength=1.5, borderpad=.25)
ax_t.text(.025, .93, '(a)', transform=ax_t.transAxes, va='top', fontsize=16,
          fontweight='bold')

# --- (c) [Fe/H] marginal ------------------------------------------------------
gf = np.linspace(*FRNG, 400)
for lab, m, c in POPS:
    ax_r.plot(kde(d['feh'][m & fin], gf), gf, color=c, lw=2.4)
    ax_r.axhline(np.nanmedian(d['feh'][m & fin]), color=c, lw=1.1)
ax_r.set(xlim=(0, None), xlabel='density')
ax_r.tick_params(labelleft=False)
ax_r.text(.07, .05, '(c)', transform=ax_r.transAxes, fontsize=16, fontweight='bold')

# --- (d) radial action --------------------------------------------------------
gj = np.linspace(*JRNG, 500)
for lab, m, c in POPS:
    a = kde(Jr[m], gj, lo=0.)
    if a is not None:
        ax_b.plot(gj, a, color=c, lw=1.6, ls='--', alpha=.9,
                  label=f'{lab}, all ({m.sum():,})')
    b = kde(Jr[m & sun], gj, lo=0.)
    if b is not None:
        ax_b.plot(gj, b, color=c, lw=3.8,
                  label=f'{lab}, solar neighbourhood ({(m & sun).sum():,})')
        ax_b.axvline(np.nanmedian(Jr[m & sun]), color=c, lw=1.1)
ax_b.set(xlim=JRNG, ylim=(0, None), xlabel=r'$J_R$ [kpc km s$^{-1}$]', ylabel='density')
ax_b.legend(loc='upper right', handlelength=2.0, borderpad=.3, labelspacing=.4, ncol=1)
ax_b.text(.025, .95, '(d)', transform=ax_b.transAxes, va='top', fontsize=16,
          fontweight='bold')

for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/au18_rbirth_feh_jr.{ext}', bbox_inches='tight')
for lab, m, c in POPS:
    print(f'  {lab:10s} median R_birth {np.nanmedian(d["R_birth"][m]):5.2f} kpc, '
          f'[Fe/H] {np.nanmedian(d["feh"][m]):+.2f}, J_R all {np.nanmedian(Jr[m]):6.0f}, '
          f'solar {np.nanmedian(Jr[m & sun]):6.0f}')
print(f'\nsaved {OUT}/au18_rbirth_feh_jr.pdf and .png')
