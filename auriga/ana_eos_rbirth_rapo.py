"""Where the born-cold mock-Eos stars were born, against where their orbits reach today.

The born-cold population of au18_rbirth_feh_jr: merger-window stars (t_form =
4.99-6.54 Gyr) that pass the Eos cut at z = 0 (|v_phi| < 80, ecc > 0.6) and were
born on a cold disc orbit (v_phi,birth >= 150 km/s).  3,300 stars.

The question the plot answers: when the merger heated these disc stars onto
radial orbits, did it also move them outwards?  R_apo > R_birth means the star
now reaches beyond where it formed; R_apo ~ R_birth means it was made radial in
place, trading circular motion for radial motion at roughly fixed apocentre.

r_apo is read from out/z0_insitu_catalog.npz, not recomputed: it comes from the
roots of Phi_eff(r) = E in the spherically averaged potential (orbit_tools.
apo_peri), which is the same approximation used for the APOGEE orbits, so the
simulated and observed r_apo mean the same thing.  It is not an orbit
integration, and for a flattened potential it is approximate -- see orbit_tools.

The born-cold map is COLUMN-NORMALISED: each R_birth column is divided by its own
total, so the colour is the distribution of R_apo *at that birth radius*, not the
abundance of stars there.  Without it the panel is dominated by the R_birth = 3-5
kpc columns, where most of the population lives, and the behaviour of the sparse
outer columns -- which is the interesting part -- is invisible.  Columns holding
fewer than NMIN_COL stars are blanked rather than shown as noise; the number
blanked and the mass they hold are printed.

born-hot is not drawn: this panel is about the born-cold population alone.  Its
statistics are still printed for comparison.

A second figure, figures/au18_eos_rapo_dist.png, gives the r_apo distributions of
the two populations on their own -- the projection of this plane onto its y axis,
for both channels rather than born-cold alone.  It is drawn per dex on a log axis
because the two medians differ by a factor of 3 and the tails reach beyond 100
kpc; see ana_splash_rapo.py for the same caveat about apo_peri's outer root
running away for nearly unbound stars.

Writes figures/au18_eos_rbirth_rapo.png and figures/au18_eos_rapo_dist.png.
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import config_au18 as C
import eos_origins as EO

RSUN = 8.1
cCOLD, cHOT = '#1F6FB2', '#FF6347'
XR, YR = (0., 15.), (0., 15.)     # equal ranges, so the 1:1 line is the diagonal
NBIN = 50
NMIN_COL = 15                     # blank columns thinner than this

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
got = {}
for k in ('rapo', 'rperi', 'ecc', 'R'):
    v = np.full(len(d['ids']), np.nan); v[ok] = cat[k][idx]; got[k] = v
rb = d['R_birth']
COLD, HOT = d['disc_born'], d['halo_born']
print(f'matched {ok.sum():,}/{len(ok):,} merger-window stars to the z=0 catalogue')

fig, ax = plt.subplots(figsize=(7.8, 6.6))
g = COLD & np.isfinite(rb) & np.isfinite(got['rapo'])
h, xe, ye = np.histogram2d(rb[g], got['rapo'][g], bins=(NBIN, NBIN), range=[XR, YR])
col = h.sum(1)
Hn = np.divide(h, col[:, None], out=np.full_like(h, np.nan),
               where=(col[:, None] >= NMIN_COL))
pcm = ax.pcolormesh(xe, ye, np.where(Hn > 0, Hn, np.nan).T, cmap='Blues', vmin=0,
                    vmax=float(np.nanpercentile(Hn, 99.5)), rasterized=True)
cb = fig.colorbar(pcm, ax=ax, pad=.015, fraction=.045, extend='max')
cb.set_label(r'fraction of the born-cold stars in each $R_{\rm birth}$ column',
             fontsize=11.5)
thin = (col > 0) & (col < NMIN_COL)
print(f'column-normalised: {NBIN} columns of {(xe[1] - xe[0]):.2f} kpc; '
      f'{int((col >= NMIN_COL).sum())} kept, {int(thin.sum())} blanked as thin '
      f'(holding {int(col[thin].sum())} stars, '
      f'{100 * col[thin].sum() / max(col.sum(), 1):.1f}% of those in range)')

gh = HOT & np.isfinite(rb) & np.isfinite(got['rapo'])

# The two reference lines that make the panel readable.
ax.plot(XR, XR, color='.25', lw=1.4, ls='--', label=r'$R_{\rm apo}=R_{\rm birth}$')

# Running median of R_apo in bins of R_birth, where there are enough stars.
edges = np.arange(0., 14.1, 1.)
ctr, med = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = g & (rb >= lo) & (rb < hi)
    if m.sum() >= 25:
        ctr.append(.5 * (lo + hi)); med.append(np.nanmedian(got['rapo'][m]))
ax.plot(ctr, med, 'o-', color=cCOLD, lw=2.6, ms=5, mec='w', mew=.8,
        label='born-cold, median $R_{\\rm apo}$')

ax.set(xlim=XR, ylim=YR, xlabel=r'$R_{\rm birth}$ [kpc]',
       ylabel=r'$R_{\rm apo}$ today [kpc]', aspect='equal')
ax.set_title('Au18 mock Eos, born-cold: birth radius against present-day apocentre\n'
             r'colour = distribution of $R_{\rm apo}$ at fixed $R_{\rm birth}$ '
             '(each column sums to 1)', fontsize=12)
ax.legend(loc='upper left', handlelength=2.0)

fig.tight_layout()
os.makedirs(C.FIG_DIR, exist_ok=True)
out = C.FIG_DIR + '/au18_eos_rbirth_rapo.png'
fig.savefig(out, bbox_inches='tight')

# --- second figure: the r_apo distributions ----------------------------------
fig2, ax2 = plt.subplots(figsize=(8.2, 5.2))
LEDGE = np.linspace(np.log10(1.5), np.log10(300.), 55)
EDGE = 10 ** LEDGE
for lab, m, c in (('born-cold', g, cCOLD), ('born-hot', gh, cHOT)):
    ra = got['rapo'][m]
    # Per dex, dN/dlog10(r): with log-spaced bins a density in linear r would put
    # the peak where the bins are narrow rather than where the stars are.
    hh = np.histogram(np.log10(ra), bins=LEDGE, density=True)[0]
    ax2.step(EDGE, np.concatenate([hh[:1], hh]), color=c, lw=2.6,
             label=f'{lab} ({m.sum():,}), median {np.nanmedian(ra):.1f} kpc')
    ax2.axvline(np.nanmedian(ra), color=c, lw=1.2, ls='--')
ax2.set(xscale='log', xlim=(1.5, 300.), ylim=(0, None), xlabel=r'$R_{\rm apo}$ [kpc]',
        ylabel=r'density per dex, $dN/d\log_{10} R_{\rm apo}$')
ax2.set_xticks([2, 5, 10, 20, 50, 100, 300])
ax2.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax2.set_title('Au18 mock Eos: present-day apocentres of the two birth channels',
              fontsize=12.5)
ax2.legend(loc='upper right', handlelength=1.8)
fig2.tight_layout()
out2 = C.FIG_DIR + '/au18_eos_rapo_dist.png'
fig2.savefig(out2, bbox_inches='tight')

# ------------------------------------------------------------------- numbers --
for lab, m in (('born-cold', g), ('born-hot', gh)):
    ra, r0 = got['rapo'][m], rb[m]
    q = lambda v: ' '.join(f'{x:6.2f}' for x in np.nanpercentile(v, [16, 50, 84]))
    print(f'\n{lab} ({m.sum():,})')
    print(f'  R_birth  16/50/84 = {q(r0)} kpc')
    print(f'  R_apo    16/50/84 = {q(ra)} kpc')
    print(f'  R_peri   16/50/84 = {q(got["rperi"][m])} kpc')
    print(f'  R_apo/R_birth median {np.nanmedian(ra / r0):.2f}; '
          f'{100 * np.mean(ra > r0):.0f}% have R_apo > R_birth')
    print(f'  reach the solar circle (R_apo > {RSUN}): {100 * np.mean(ra > RSUN):.1f}%; '
          f'R_apo > 20 kpc: {100 * np.mean(ra > 20):.1f}%; '
          f'above the panel (R_apo > {YR[1]:.0f}): {100 * np.mean(ra > YR[1]):.1f}%')
print('\nsaved', out)
print('saved', out2)
