"""Publication figure: the orbits Au18 stars are born on, through the GS/E merger.

eps = L_z/L_circ(E) and z_max are measured in the first stored snapshot at or
after each star formed, in that epoch's own AGAMA CylSpline potential and disc
frame.  See METHOD_zmax_from_Jz.md for the z_max approximation and its accuracy.

  disc-born   eps > 0.8  OR  z_max < 1.5 kpc
  halo-born   eps <= 0.8 AND z_max >= 1.5 kpc

Densities are Gaussian kernel estimates.  With 1.9M stars a direct gaussian_kde
evaluation is needlessly slow, so each is computed by convolving a fine weighted
histogram with the Gaussian kernel, which is the same estimator on a grid.  The
eps kernel is reflected at eps = 1, the hard upper bound, so no density leaks
past it.

Writes Fig_paper/au18_birth_orbits.pdf and .png.
"""
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib as mpl
import matplotlib.pyplot as plt
import config_au18 as C

OUT = '/data/hz420-2/EoS_explore/Fig_paper'
os.makedirs(OUT, exist_ok=True)
CUT, ZCUT = 0.8, 1.5
# Kernel widths (Gaussian sigma, NOT a bin width -- sigma = 0.15 has FWHM 0.35 Gyr
# and oversmooths the narrow halo-born spike while leaving the broad disc-born SFR
# alone, which drags the ratio in (d) down from 0.54 to 0.35).  sigma = 0.05 Gyr
# reproduces the 0.15 Gyr histogram to 0.2 per cent.  Going finer is not justified:
# eps and z_max are measured once per snapshot, so the classification itself is only
# resolved to the ~0.15 Gyr snapshot spacing.
BW_T, BW_E = 0.08, 0.010            # kernel widths: Gyr, and in eps
T_PERI, T_SPIN = 5.0, 3.4
# halo-born is tomato throughout, in this figure and in au18_birth_positions, so
# the two read as one set.  That rules warm reds out for the GS/E marker, which is
# violet -- high contrast against both the tomato and the teal spin-up line.
EPOCHS = [('before', 3.5, 4.7, '#5E60CE'), ('during', 4.9, 5.7, '#FF6347'),
          ('after', 6.6, 8.0, '#2E7D32')]
cD, cH, cT, cM, cS = '#1F6FB2', '#FF6347', '#2B2B2B', '#8E24AA', '#00897B'

# Times New Roman is not installed here; Nimbus Roman and Liberation Serif are
# metric-compatible clones of it, so the output is Times either way.  Times New
# Roman is listed first so it wins if the real font is ever installed.
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
    'xtick.minor.size': 2.8, 'ytick.minor.size': 2.8,
    'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})

a = np.load(C.OUT_DIR + '/birth_orbits_actions.npz')
zx = np.load(C.OUT_DIR + '/birth_orbits_zmax.npz')
q = np.load(C.OUT_DIR + '/insitu_imass.npz')
tf, eb, mi, zm = a['tform'], a['eps_birth'], q['imass'], zx['zmax_birth']
g = np.isfinite(eb) & np.isfinite(mi) & np.isfinite(zm)
tf, eb, mi, zm = tf[g], eb[g], mi[g], zm[g]
disc = (eb > CUT) | (zm < ZCUT)
halo = ~disc
TMIN = np.floor(tf.min() * 10) / 10
print(f'{g.sum():,} in-situ stars; halo-born {100 * halo.mean():.1f}%')


def sfr_kde(t, w, grid, bw=BW_T):
    """SFR [Msun/yr] as a Gaussian kernel density in cosmic time."""
    fine = np.arange(grid[0] - 8 * bw, grid[-1] + 8 * bw, bw / 8.)
    h = np.histogram(t, bins=fine, weights=w)[0]
    sm = gaussian_filter1d(h, 8.) / (bw / 8.)          # -> Msun per Gyr
    ctr = .5 * (fine[:-1] + fine[1:])
    return np.interp(grid, ctr, sm) / 1e9


def eps_kde(e, w, grid, bw=BW_E, hi=1.0):
    """Density in eps, with the kernel reflected at the hard upper bound eps = hi."""
    fine = np.arange(-1.4, hi + bw / 8., bw / 8.)
    h = np.histogram(e, bins=fine, weights=w)[0]
    h = h + h[::-1] * 0                                  # keep shape explicit
    pad = np.concatenate([h, h[::-1]])                   # mirror about the top edge
    sm = gaussian_filter1d(pad, 8.)[:len(h)]
    ctr = .5 * (fine[:-1] + fine[1:])
    sm = sm / (sm.sum() * (bw / 8.))
    return np.interp(grid, ctr, sm)


fig, axes = plt.subplots(2, 2, figsize=(11.6, 6.5))
tgrid = np.linspace(TMIN, C.T0_GYR, 500)
egrid = np.linspace(-0.2, 1.0, 400)


def markers(ax, lab=False):
    ax.axvline(T_SPIN, color=cS, ls=':', lw=2.6, zorder=1,
               label='disc spin-up' if lab else None)
    ax.axvline(T_PERI, color=cM, ls='--', lw=2.4, zorder=1,
               label='GS/E pericentre' if lab else None)


# (a) birth circularity against time ------------------------------------------
ax = axes[0, 0]
tb = np.arange(TMIN, C.T0_GYR + .15, .15)
ebn = np.linspace(-1, 1, 81)
H, _, _ = np.histogram2d(tf, eb, bins=[tb, ebn], weights=mi)
col = H.sum(1)
Hn = np.divide(H, col[:, None], out=np.full_like(H, np.nan), where=col[:, None] > 0)
pc = ax.pcolormesh(tb, ebn, Hn.T, cmap='Blues', vmin=0,
                   vmax=float(np.nanpercentile(Hn, 98)), rasterized=True)
ax.axhline(CUT, color='.25', ls='--', lw=.9)
markers(ax)
for t, c, txt in [(T_SPIN, cS, 'disc spin-up'), (T_PERI, cM, 'GS/E pericentre')]:
    ax.text(t - .22, .05, txt, transform=ax.get_xaxis_transform(), rotation=90,
            va='bottom', ha='right', fontsize=12, color=c)
cb = fig.colorbar(pc, ax=ax, pad=.02, extend='max', fraction=.05)
cb.set_label('fraction of mass formed', fontsize=13)
cb.ax.tick_params(labelsize=12)
ax.set(xlim=(TMIN, C.T0_GYR), ylim=(-1, 1), xlabel='cosmic time [Gyr]',
       ylabel=r'$\epsilon$ at birth', xticks=np.arange(2, 14, 2))
ax.text(.04, .05, '(a)', fontsize=16, transform=ax.transAxes, va='bottom', fontweight='bold')

# (b) birth circularity, three epochs -----------------------------------------
ax = axes[0, 1]
for lab, lo, hi, c in EPOCHS:
    m = (tf >= lo) & (tf < hi)
    ax.plot(egrid, eps_kde(eb[m], mi[m], egrid), color=c, lw=1.8,
            label=f'{lab} ({lo}–{hi} Gyr)')
ax.axvline(CUT, color='.35', ls='--', lw=.9)
ax.set(xlim=(-.2, 1), ylim=(0, None), xlabel=r'$\epsilon$ at birth',
       ylabel='normalised density (mass-weighted)')
ax.legend(loc='upper left', handlelength=1.6, borderpad=.25, labelspacing=.35)
ax.text(.96, .94, '(b)', fontsize=16, transform=ax.transAxes, va='top', ha='right', fontweight='bold')

# (c) star-formation history, split -------------------------------------------
ax = axes[1, 0]
st = sfr_kde(tf, mi, tgrid)
sd = sfr_kde(tf[disc], mi[disc], tgrid)
sh = sfr_kde(tf[halo], mi[halo], tgrid)
markers(ax, lab=True)
ax.plot(tgrid, st, color=cT, lw=1.6, label='total')
ax.plot(tgrid, sd, color=cD, lw=1.6, label='disc-born')
ax.fill_between(tgrid, 0, sd, color=cD, alpha=.15, lw=0)
ax.plot(tgrid, sh, color=cH, lw=1.6, label='halo-born')
ax.fill_between(tgrid, 0, sh, color=cH, alpha=.22, lw=0)
ax.set(xlim=(TMIN, C.T0_GYR), ylim=(0, 1.06 * st.max()), xlabel='cosmic time [Gyr]',
       ylabel=r'SFR [M$_\odot$ yr$^{-1}$]', xticks=np.arange(2, 14, 2))
ax.legend(loc='upper right', handlelength=1.6, borderpad=.25, labelspacing=.35, ncol=2)
ax.text(.04, .94, '(c)', fontsize=16, transform=ax.transAxes, va='top', fontweight='bold')

# (d) the ratio ----------------------------------------------------------------
ax = axes[1, 1]
rat = np.divide(sh, sd, out=np.full_like(sh, np.nan), where=sd > 1.)
markers(ax)
ax.plot(tgrid, rat, color=cH, lw=2.0)
# The full range is shown: the rise before t ~ 2.5 Gyr is the pre-disc era, when
# there is barely a disc to normalise against, and clipping it would hide that.
ax.set(xlim=(TMIN, C.T0_GYR), ylim=(0, 1.05 * np.nanmax(rat)),
       xlabel='cosmic time [Gyr]',
       ylabel='halo-born / disc-born SFR', xticks=np.arange(2, 14, 2))
ax.text(.04, .94, '(d)', fontsize=16, transform=ax.transAxes, va='top', fontweight='bold')

fig.tight_layout(pad=.6, w_pad=1.8, h_pad=1.0)
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/au18_birth_orbits.{ext}', bbox_inches='tight')

pk = np.nanargmax(np.where(tgrid > 2.5, rat, np.nan))
print(f'peak halo/disc ratio {rat[pk]:.2f} at t = {tgrid[pk]:.2f} Gyr; '
      f'halo-born SFR there {sh[pk]:.2f}, total {st[pk]:.2f} Msun/yr')
for lab, lo, hi, _ in EPOCHS:
    m = (tf >= lo) & (tf < hi)
    print(f'  {lab:7s} {lo}-{hi} Gyr: halo-born {100 * mi[m & halo].sum() / mi[m].sum():5.1f}%'
          f'   halo/disc {mi[m & halo].sum() / mi[m & disc].sum():.3f}')
print(f'\nsaved {OUT}/au18_birth_orbits.pdf and .png')
