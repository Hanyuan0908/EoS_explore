"""Star-forming disc SIZE against time in Auriga halo 18: R_d and R_1/2.

The scale-length cousin of the Au18 panels of ana_gas_disc_compare.py, which
measures the disc by half-mass radius alone.  Same data, same merger epochs, same
normalisation.  Two things are added.

**Two size measures.**

  R_1/2   half-mass radius -- the enclosed-mass radius, set by the whole profile
  R_d     exponential scale length -- fitted to ln Sigma(R) over
          [0.5, 2.5] x R_1/2, so it describes the disc's outer slope and ignores
          the centre entirely

A merger redistributes mass, and the two respond differently: a central
concentration pulls R_1/2 in without touching the outer disc, while R_d is blind
to it.  Where they disagree the profile is not a single exponential, and the
disagreement is the result, not a fault in either measure.

**Two tracers.**

  star-forming gas   cells with SFR > 0 -- where stars are *eligible* to form
  new stars          t_snap - t_form < 100 Myr -- where they *did* form

The second is the stellar counterpart of the first and the closer analogue of an
observed young disc.  Where the gas disc is larger than the new-star disc, gas is
eligible over a wider area than it is actually using.

  left   both tracers against cosmic time, R_d solid and R_1/2 dashed, plus R_90
         of the star-forming gas dotted -- the radius enclosing 90 per cent of it,
         which is where the eligible-but-unused outer gas shows up: R_90 tracks
         the far outskirts, R_1/2 the bulk, R_d only the fitted slope
  right  the same aligned on coalescence and normalised to their own pre-merger
         values -- the counterpart of the lower-left panel of gas_disc_vs_merger

R_d and the Sigma(R) profiles behind it come from auriga/prep_gas_disc_au18.py;
the profiles are stored, so the fit range and the quality threshold can be
changed here without another pass over the snapshots.  R_d points whose fit has
an rms residual above RMS_MAX are drawn hollow: through the merger the profiles
are clean exponentials, but late on they flatten into a broad ring and the
straight line is then fitted to something that is not a disc profile.
auriga/diag_sigma_exp_fit.py plots the fits themselves.

Writes figures_sim/gas_disc_scalelength_au18.png.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = ROOT + '/figures_sim'
os.makedirs(OUT, exist_ok=True)

A = np.load(ROOT + '/auriga/out/gas_disc_evolution_au18.npz')
t = A['time']
RMS_MAX = 0.25          # dex; above this the profile is not an exponential
EVENTS = [(3.25, 'first apo'), (5.0, 'plunge'), (5.4, 'coalescence')]
TREF, TMAX = 5.4, 14.

TRACERS = [
    dict(name='star-forming gas', colour='#b2182b',
         rd=A['rd_sf'], rh=A['rhalf_sf'], sig=A['sigma_sf']),
    dict(name=f"new stars ($<{A['new_gyr'] * 1000:.0f}$ Myr)", colour='#5E35B1',
         rd=A['rd_new'], rh=A['rhalf_new'], sig=A['sigma_new']),
]


def fit_rms(sig, rhalf, edges=A['rbins'], rng=A['fit_range']):
    """Rms residual [dex] of the exponential fit that produced R_d."""
    lo, hi = rng
    ctr = .5 * (edges[:-1] + edges[1:])
    ok = (ctr > lo * rhalf) & (ctr < hi * rhalf) & (sig > 0)
    if ok.sum() < 5:
        return np.nan
    c = np.polyfit(ctr[ok], np.log(sig[ok]), 1)
    return float(np.sqrt(np.mean((np.log10(np.exp(np.polyval(c, ctr[ok])))
                                  - np.log10(sig[ok])) ** 2)))


for T in TRACERS:
    T['rms'] = np.array([fit_rms(s, r) for s, r in zip(T['sig'], T['rh'])])
    T['good'] = T['rms'] <= RMS_MAX

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

# --- left: through cosmic time ------------------------------------------------
# Colour is the tracer, line style is the measure: solid R_d, dashed R_1/2.
ax = axes[0]
for T in TRACERS:
    c, g = T['colour'], T['good']
    ax.plot(t, T['rd'], '-', color=c, lw=2.2, label=f"{T['name']}, $R_d$")
    ax.plot(t[g], T['rd'][g], 'o', color=c, ms=4.5)
    ax.plot(t[~g], T['rd'][~g], 'o', mfc='white', mec=c, mew=1.2, ms=4.5)
    ax.plot(t, T['rh'], '--', color=c, lw=1.3, alpha=.65,
            label=f"{T['name']}, $R_{{1/2}}$")
ax.plot(t, A['r90_sf'], ':', color=TRACERS[0]['colour'], lw=1.6, alpha=.8,
        label='star-forming gas, $R_{90}$')
ax.plot(t, A['rd_star'], '-', color='.45', lw=1.4, label='all stars, $R_d$')
ymax = np.nanmax(A['r90_sf']) * 1.10
for x, lab in EVENTS:
    ax.axvline(x, color='k', lw=1.1, ls='--')
    # White bbox: the event lines fall where the curves run, so the labels sit on
    # top of data whatever height they are given.
    ax.text(x - .10, .015 * ymax, lab, rotation=90, ha='right', va='bottom',
            fontsize=8.5, bbox=dict(fc='white', ec='none', alpha=.85, pad=1.5))
ax.set(xlim=(0, TMAX), ylim=(0, ymax), xlabel='cosmic time [Gyr]',
       ylabel='disc size [kpc]')
ax.set_title('Auriga halo 18\n' + r'$n_H>0.13$ cm$^{-3}$ (SFR $>0$) vs stars formed '
             'in the last 100 Myr', fontsize=11)
# Two columns, not one: a single column reaches down to y ~ 12 kpc and hides the
# early R_90 peak, which is the one curve that goes up there.
ax.legend(fontsize=8, loc='upper left', framealpha=.95, ncol=2)

# --- right: aligned on coalescence, normalised to pre-merger ------------------
ax = axes[1]
dt = t - TREF
pre = (dt > -2.0) & (dt < -1.0)
for T in TRACERS:
    c, g = T['colour'], T['good']
    for y, sty, lw, lab in [(T['rd'], '-', 2.2, '$R_d$'),
                            (T['rh'], '--', 1.5, '$R_{1/2}$')]:
        ref = np.nanmean(y[pre])
        ax.plot(dt, y / ref, sty, color=c, lw=lw, alpha=1. if sty == '-' else .65,
                label=f"{T['name']}, {lab}  (ref $={ref:.2f}$ kpc)")
        T['ref_' + lab] = ref
    ax.plot(dt[g], (T['rd'] / T['ref_$R_d$'])[g], 'o', color=c, ms=4)
    ax.plot(dt[~g], (T['rd'] / T['ref_$R_d$'])[~g], 'o', mfc='white', mec=c,
            mew=1.2, ms=4)
ax.axvline(0, color='k', lw=1.2, ls='--')
ax.axhline(1, color='.6', lw=.8, ls=':')
ax.set(xlim=(-3, 6), xlabel='time since coalescence [Gyr]',
       ylabel='size / pre-merger value')
ax.set_title('Aligned on coalescence, each normalised to its own pre-merger value',
             fontsize=10.5)
ax.legend(fontsize=8.5)

fig.suptitle('Auriga halo 18: how the star-forming disc contracts through the merger, '
             'by scale length and by half-mass radius', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, .93])
out = OUT + '/gas_disc_scalelength_au18.png'
fig.savefig(out, dpi=150)

# ------------------------------------------------------------------- numbers --
win = (dt > -2.0) & (dt < 0.5)
post = (dt > 1.0) & (dt < 3.0)
for T in TRACERS:
    print(f"\n{T['name']}")
    for y, lab in [(T['rd'], 'R_d  '), (T['rh'], 'R_1/2')]:
        i = int(np.nanargmin(np.where(win, y, np.inf)))
        print(f'  {lab}  pre-merger {np.nanmean(y[pre]):5.2f} kpc   '
              f'minimum {y[i]:5.2f} at t = {t[i]:5.2f} (dt = {dt[i]:+.2f}) '
              f'-> {100 * (1 - y[i] / np.nanmean(y[pre])):3.0f}% contraction   '
              f'1-3 Gyr on {np.nanmean(y[post]):5.2f} '
              f'({np.nanmean(y[post]) / np.nanmean(y[pre]):.2f}x)')
    print(f"  R_1/2 / R_d = {np.nanmedian(T['rh'] / T['rd']):.2f} median "
          f'(1.68 for a pure exponential)')
    for a, b in [(0, 6), (6, 10), (10, 14)]:
        m = (t >= a) & (t < b)
        print(f"  fit rms t = {a:2d}-{b:2d} Gyr: median {np.nanmedian(T['rms'][m]):.3f} dex, "
              f"max {np.nanmax(T['rms'][m]):.3f}  "
              f"({(~T['good'][m]).sum()}/{m.sum()} above {RMS_MAX})")
r90, rh = A['r90_sf'], A['rhalf_sf']
i90 = int(np.nanargmin(np.where(win, r90, np.inf)))
print(f"\nstar-forming gas R_90: pre-merger {np.nanmean(r90[pre]):5.2f} kpc   "
      f"minimum {r90[i90]:5.2f} at t = {t[i90]:5.2f} (dt = {dt[i90]:+.2f}) "
      f"-> {100 * (1 - r90[i90] / np.nanmean(r90[pre])):3.0f}% contraction   "
      f"1-3 Gyr on {np.nanmean(r90[post]):5.2f} "
      f"({np.nanmean(r90[post]) / np.nanmean(r90[pre]):.2f}x)")
print(f"  R_90 / R_1/2 = {np.nanmedian(r90 / rh):.2f} median "
      f"(1.55 for a pure exponential)")
print(f"\nnew stars per snapshot: {A['n_new'].min():,} to {A['n_new'].max():,} particles")
print('saved', out)
