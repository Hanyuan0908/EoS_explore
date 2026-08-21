"""Birth-time / age distributions of the two Au18 Eos channels.

NOTE the sample is merger-born by construction (t_birth = 4.99-6.54 Gyr), so these
are distributions *within* the merger window, not full star-formation histories.
The question is whether the two channels are active at the same times.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import config_au18 as C
import channels_au18 as ch_mod

os.makedirs(C.FIG_DIR, exist_ok=True)

d = ch_mod.load()
zb = d['z_birth']; tb = d['tform']
age = C.T0_GYR - tb
base, A, B = d['base'], d['A'], d['B']
cA, cB, cP = '#2166ac', '#b2182b', '.55'
LA, LB = d['label_A'], d['label_B']
LP = f'all merger-born (N={base.sum():,})'
T_COAL = 5.4

print(f'window: t_birth {tb[base].min():.2f}-{tb[base].max():.2f} Gyr '
      f'(lookback age {age[base].max():.2f}-{age[base].min():.2f} Gyr)')
print(f'{"":26s} {"A heated":>9s} {"B born-rad":>12s} {"all":>10s}')
for lab, f in [('median t_birth [Gyr]', np.median),
               ('25th pct t_birth    ', lambda x: np.percentile(x, 25)),
               ('75th pct t_birth    ', lambda x: np.percentile(x, 75))]:
    print(f'{lab:26s} {f(tb[A]):9.2f} {f(tb[B]):12.2f} {f(tb[base]):10.2f}')
print(f'{"median age [Gyr]":26s} {np.median(age[A]):9.2f} {np.median(age[B]):12.2f} '
      f'{np.median(age[base]):10.2f}')
for lab, m in [('frac born before coal.', tb < T_COAL), ('frac born after coal. ', tb >= T_COAL)]:
    print(f'{lab:26s} {m[A].mean():9.3f} {m[B].mean():12.3f} {m[base].mean():10.3f}')
ks = ks_2samp(tb[A], tb[B])
print(f'\nKS(t_birth A vs B): D={ks.statistic:.3f} p={ks.pvalue:.2g}')
print(f'median offset B-A = {np.median(tb[B]) - np.median(tb[A]):+.3f} Gyr '
      f'(B is {"older" if np.median(tb[B]) < np.median(tb[A]) else "younger"})')

# ------------------------------------------------------------------ figure --
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.6))
bins = np.linspace(tb[base].min(), tb[base].max(), 41)
ctr = .5 * (bins[:-1] + bins[1:])


def age_axis(ax):
    """Secondary axis in lookback age, since that is what observations measure."""
    sec = ax.secondary_xaxis('top', functions=(lambda t: C.T0_GYR - t,
                                               lambda a: C.T0_GYR - a))
    sec.set_xlabel('lookback age [Gyr]')


ax = axes[0, 0]
ax.hist(tb[base], bins=bins, density=True, histtype='stepfilled', color=cP, alpha=.28, label=LP)
for m, c, l in [(A, cA, LA), (B, cB, LB)]:
    ax.hist(tb[m], bins=bins, density=True, histtype='step', lw=2, color=c, label=l)
    ax.axvline(np.median(tb[m]), color=c, ls=':', lw=1.4)
ax.axvline(T_COAL, color='goldenrod', lw=2, alpha=.75, label='coalescence')
ax.set(xlabel='birth cosmic time [Gyr]', ylabel='normalised density',
       title='Birth time (dotted = median)')
ax.legend(fontsize=8.5); age_axis(ax)

ax = axes[0, 1]
for m, c, l in [(base, cP, LP), (A, cA, LA), (B, cB, LB)]:
    xs = np.sort(tb[m]); ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=c, lw=2, label=l)
ax.axvline(T_COAL, color='goldenrod', lw=2, alpha=.75)
ax.axhline(.5, color='k', lw=.6, ls='--')
ax.set(xlabel='birth cosmic time [Gyr]', ylabel='cumulative fraction', ylim=(0, 1),
       title=f'Cumulative: KS D={ks.statistic:.2f}')
ax.legend(fontsize=8.5, loc='lower right'); age_axis(ax)

# Raw counts: the actual formation history of each channel, not renormalised.
ax = axes[1, 0]
for m, c, l in [(A, cA, LA), (B, cB, LB)]:
    n, _ = np.histogram(tb[m], bins=bins)
    ax.step(ctr, n, where='mid', color=c, lw=2, label=l)
ax.axvline(T_COAL, color='goldenrod', lw=2, alpha=.75, label='coalescence')
ax.set(xlabel='birth cosmic time [Gyr]', ylabel='stars per bin',
       title='Raw counts (channel formation history)')
ax.legend(fontsize=8.5); age_axis(ax)

# What fraction of all merger-born stars ends up in each channel, vs birth time.
ax = axes[1, 1]
ntot, _ = np.histogram(tb[base], bins=bins)
for m, c, l in [(A, cA, LA), (B, cB, LB)]:
    n, _ = np.histogram(tb[m], bins=bins)
    good = ntot > 50
    ax.plot(ctr[good], 100 * n[good] / ntot[good], color=c, lw=2, marker='o', ms=3.5, label=l)
ax.axvline(T_COAL, color='goldenrod', lw=2, alpha=.75)
ax.set(xlabel='birth cosmic time [Gyr]', ylabel='per cent of merger-born stars',
       title='Channel share of star formation')
ax.legend(fontsize=8.5); age_axis(ax)

fig.suptitle('Au18 Eos channels: birth-time distributions within the merger window')
fig.tight_layout(rect=[0, 0, 1, .93])
out = C.FIG_DIR + '/au18_eos_channels_birth_time.png'
fig.savefig(out, dpi=150)
np.savez(C.OUT_DIR + '/eos_channels_birth_time.npz',
         t_birth_A=tb[A], t_birth_B=tb[B], t_birth_all=tb[base], ks_D=ks.statistic)
print('\nsaved', out)
