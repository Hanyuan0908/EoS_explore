"""The halo-born age distribution of au18_birth_orbits panel (a), before and after
a 30 per cent age error.

Sample: the halo-born stars of that panel -- in-situ, with a measured birth orbit,
eps <= 0.8 AND z_max >= 1.5 kpc -- weighted by GFM_InitialMass.  Stars older than
AGE_MAX are dropped **before** the smearing, on their true age: they formed before
the disc spun up at t = 3.4 Gyr and in a real survey are separable on chemistry
(metal-poor, Aurora-like) rather than on the age itself.  That is the optimistic
version of the cut -- made on the observed age instead it would scatter
merger-epoch stars out and old stars in, and would clean the sample less well.

Two error models, selected on the command line:

  (default)   Gaussian in age, sigma = FRAC * age, the literal reading of
              sigma_age/age = 0.3: a 2 Gyr star is smeared by 0.6 Gyr and a 9 Gyr
              star by 2.7 Gyr.
  `log 0.05`  Gaussian in log10 age with sigma = SIG_LOG dex, i.e. a log-normal
              in age.  This is what BINGO actually produces and how Ciuca et al.
              (2024, MNRAS 528, L122) quote their precision: they keep stars with
              sigma_log tau <= 0.2 dex and quote 0.05 dex as the typical value.
              By their own conversion, sigma_tau = 0.5 (10^(log tau + sigma) -
              10^(log tau - sigma)), 0.05 dex is 11.5 per cent and 0.2 dex is 48
              per cent -- so the cut is permissive and the typical star is far
              better than it.

The log-normal is the more realistic model: it is asymmetric in linear age, which
a Gaussian is not, and it cannot scatter a star to a negative age.

Applied as a variable-width convolution rather than by Monte Carlo: the true ages
are binned at 0.02 Gyr, each bin is spread by its own Gaussian, and the bins are
summed.  That is the infinite-sample limit of drawing an error per star, so the
smeared curve carries no sampling noise of its own.  Both curves are normalised
to the same mass, so the smearing moves mass around rather than adding any.

The median and the mode of each curve are marked.  They answer different
questions and move differently under the smearing: the median is an integral over
the whole distribution and barely notices a symmetric kernel, while the mode is a
local property of the peak and follows it.  The true mode is taken from the
displayed (0.1 Gyr smoothed) curve, since the mode of a raw 0.02 Gyr histogram is
set by shot noise; the medians are taken from the unsmoothed weights.

The true curve is drawn with a 0.1 Gyr Gaussian smoothing -- display only, and
27x narrower than the error at the merger epoch.  Without it the fine bins the
convolution needs read as noise next to the smeared curve.

Observed ages are NOT clipped at the age of the universe; real catalogues truncate
there, piling that weight up at the boundary instead.

Writes figures/au18_age_error_smearing.png.
"""
import os, sys
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib as mpl
import matplotlib.pyplot as plt
import config_au18 as C

# `python diag_age_error_smearing.py log [sigma_dex]` for the log-normal model.
KIND = 'log' if 'log' in sys.argv[1:] else 'frac'
SIG_LOG = float(sys.argv[2]) if KIND == 'log' and len(sys.argv) > 2 else 0.05
FRAC = 0.30                     # sigma_age / age, the Gaussian-in-age model
TAG = '' if KIND == 'frac' else f'_siglog{SIG_LOG:g}'
# Ciuca et al.'s conversion from dex to a linear fractional error.
EQ_FRAC = .5 * (10 ** SIG_LOG - 10 ** -SIG_LOG)
LABEL = (r'smeared, $\sigma_{\rm age}/{\rm age}=%.2f$' % FRAC if KIND == 'frac'
         else r'smeared, $\sigma_{\log\tau}=%.2f$ dex $\;(\sigma_\tau/\tau=%.2f)$'
              % (SIG_LOG, EQ_FRAC))
AGE_MAX = 10.0                  # drop true ages above this (pre-spin-up stars)
CUT, ZCUT = 0.8, 1.5            # the panel (a) split, unchanged
MERGER = (4.99, 6.54)           # the standard window, in cosmic time
cH = '#FF6347'
cMED, cMODE = '#2B2B2B', '#00897B'      # median near-black, mode teal

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 13.5, 'axes.labelsize': 15,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'figure.dpi': 140, 'savefig.dpi': 200,
})

a = np.load(C.OUT_DIR + '/birth_orbits_actions.npz')
zx = np.load(C.OUT_DIR + '/birth_orbits_zmax.npz')
q = np.load(C.OUT_DIR + '/insitu_imass.npz')
tf, eb, mi, zm = a['tform'], a['eps_birth'], q['imass'], zx['zmax_birth']
g = np.isfinite(eb) & np.isfinite(mi) & np.isfinite(zm)
tf, eb, mi, zm = tf[g], eb[g], mi[g], zm[g]
halo = ~((eb > CUT) | (zm < ZCUT))
age = C.T0_GYR - tf
sel = halo & (age <= AGE_MAX)
print(f'{g.sum():,} in-situ stars, halo-born {halo.sum():,}; '
      f'after age <= {AGE_MAX} Gyr (t_form >= {C.T0_GYR - AGE_MAX:.2f} Gyr): '
      f'{sel.sum():,} stars, {mi[sel].sum():.3e} Msun '
      f'({100 * mi[sel].sum() / mi[halo].sum():.1f}% of the halo-born mass)')

FINE = np.arange(0., C.T0_GYR + .02, .02)          # true-age bins
FCTR = .5 * (FINE[:-1] + FINE[1:])
GRID = np.linspace(0., 16., 800)                   # observed-age grid
SHOW_SMOOTH = 0.1 / .02                            # display smoothing, in bins

h = np.histogram(age[sel], bins=FINE, weights=mi[sel])[0]
true = gaussian_filter1d(h, SHOW_SMOOTH) / (h.sum() * .02)
if KIND == 'frac':
    # Each true-age bin spreads by its own sigma = FRAC * age.  The zero-age bin
    # has sigma = 0 and would be a delta function, so it is floored at half a bin.
    sig = np.maximum(FRAC * FCTR, .01)
    k = np.exp(-.5 * ((GRID[:, None] - FCTR[None, :]) / sig[None, :]) ** 2)
    k /= (sig[None, :] * np.sqrt(2 * np.pi))
else:
    # Gaussian in log10 age, expressed as a density in age: the Jacobian
    # d(log10 g)/dg = 1/(g ln10) is what makes it asymmetric.  Both grids are
    # floored away from zero, where log age diverges and the density is zero.
    gg = np.maximum(GRID, 1e-3)[:, None]
    cc = np.maximum(FCTR, 1e-3)[None, :]
    k = np.exp(-.5 * ((np.log10(gg) - np.log10(cc)) / SIG_LOG) ** 2)
    k /= (gg * np.log(10.) * SIG_LOG * np.sqrt(2 * np.pi))
    k[GRID <= 0] = 0.
obs = (k * h[None, :]).sum(1) / h.sum()

def wmedian(x, w):
    """Weighted median by linear interpolation of the cumulative weight."""
    c = np.cumsum(w)
    return float(np.interp(.5 * c[-1], c, x))


MED_T, MED_O = wmedian(FCTR, h), wmedian(GRID, obs)
MODE_T, MODE_O = FCTR[np.argmax(true)], GRID[np.argmax(obs)]

fig, ax = plt.subplots(figsize=(7.6, 5.2))
AGE_MERGER = (C.T0_GYR - MERGER[1], C.T0_GYR - MERGER[0])
ax.axvspan(*AGE_MERGER, color='#8E24AA', alpha=.10, lw=0)
ax.text(np.mean(AGE_MERGER), .97, 'merger\nwindow', transform=ax.get_xaxis_transform(),
        ha='center', va='top', fontsize=11.5, color='#8E24AA')
ax.plot(FCTR, true, '--', color=cH, lw=1.8, label='true age')
ax.plot(GRID, obs, '-', color=cH, lw=2.8, label=LABEL)
# Statistic in the colour, true/smeared in the line style, matching the curves.
for x, c, ls in [(MED_T, cMED, '--'), (MED_O, cMED, '-'),
                 (MODE_T, cMODE, '--'), (MODE_O, cMODE, '-')]:
    ax.axvline(x, color=c, ls=ls, lw=1.5, alpha=.9, zorder=1)
ax.plot([], [], color=cMED, lw=1.5, label=f'median: {MED_T:.2f} $\\to$ {MED_O:.2f} Gyr')
ax.plot([], [], color=cMODE, lw=1.5, label=f'mode: {MODE_T:.2f} $\\to$ {MODE_O:.2f} Gyr')

# Where the sample ends: everything to the right of this is stars scattered past
# the cut by their own error, not a population.
ax.axvline(AGE_MAX, color='.45', ls=':', lw=1.4)
ax.text(AGE_MAX + .16, .55, f'sample cut at {AGE_MAX:.0f} Gyr',
        transform=ax.get_xaxis_transform(), rotation=90, fontsize=10.5, color='.35',
        va='center')
ax.set(xlim=(0, 15), ylim=(0, None), xlabel='age [Gyr]',
       ylabel='normalised density [Gyr$^{-1}$]')
ax.set_title('Au18 halo-born stars, true age $\\leq$ 10 Gyr', fontsize=13)
ax.legend(loc='upper left', handlelength=2.0)

fig.tight_layout()
os.makedirs(C.FIG_DIR, exist_ok=True)
out = C.FIG_DIR + f'/au18_age_error_smearing{TAG}.png'
fig.savefig(out, bbox_inches='tight')

# ------------------------------------------------------------------- numbers --
print(f'\nerror model: ' + ('Gaussian in age, sigma/age = %.2f' % FRAC if KIND == 'frac'
      else 'log-normal, sigma_log tau = %.3f dex (sigma_tau/tau = %.3f)' % (SIG_LOG, EQ_FRAC)))
print(f'merger window t = {MERGER[0]}-{MERGER[1]} Gyr  ->  age = '
      f'{AGE_MERGER[0]:.2f}-{AGE_MERGER[1]:.2f} Gyr')
print(f'  peak: true {true.max():.3f} Gyr^-1 at {FCTR[np.argmax(true)]:.2f} Gyr  ->  '
      f'smeared {obs.max():.3f} at {GRID[np.argmax(obs)]:.2f} Gyr '
      f'({100 * (1 - obs.max() / true.max()):.0f}% lower)')
for lab, x, y in [('true   ', FCTR, true), ('smeared', GRID, obs)]:
    inw = (x >= AGE_MERGER[0]) & (x <= AGE_MERGER[1])
    print(f'  {lab}: {100 * np.trapz(y[inw], x[inw]):5.1f}% of the mass falls in the '
          f'merger window')
print(f'  median: true {MED_T:.2f} -> smeared {MED_O:.2f} Gyr ({MED_O - MED_T:+.2f})')
print(f'  mode:   true {MODE_T:.2f} -> smeared {MODE_O:.2f} Gyr ({MODE_O - MODE_T:+.2f})')
print(f'  smeared mass past the cut at {AGE_MAX} Gyr: '
      f'{100 * np.trapz(obs[GRID > AGE_MAX], GRID[GRID > AGE_MAX]):.1f}%; '
      f'past the age of the universe: '
      f'{100 * np.trapz(obs[GRID > C.T0_GYR], GRID[GRID > C.T0_GYR]):.1f}%')
print('\nsaved', out)
