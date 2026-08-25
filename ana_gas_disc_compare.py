"""Star-forming gas disc size against time, in both simulations, versus the merger.

Each galaxy uses its OWN published star-formation criterion -- they differ by a
factor of ~8 in density, so a shared cut would not be meaningful:

  gastro (Clumpy+merger)  T < 1.5e4 K and n_H > 1 cm^-3
                          Amarante et al. 2022, ApJ 937, 12
  Auriga halo 18          the code's StarFormationRate flag, i.e. n_H > 0.13 cm^-3
                          on the Springel & Hernquist effective EOS
                          Grand et al. 2017, MNRAS 467, 179

Merger epochs.  gastro's dwarf is on a controlled orbit with pericentres at 1.6,
2.5 and 3.2 Gyr, the last being full disruption.  Au18's were measured from the
simulation (auriga/PROGRESS.md): first apocentre ~3.25 Gyr, pericentre plunge
~5.0, coalescence ~5.4.  The bottom row aligns the two on their final event
(3.2 and 5.4 Gyr) and normalises each disc to its own pre-merger size, so the
trends can be compared despite the galaxies differing in size and epoch.

Writes figures_sim/gas_disc_vs_merger.png.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = ROOT + '/figures_sim'
os.makedirs(OUT, exist_ok=True)

G = np.load(ROOT + '/gastro/out/gas_disc_evolution.npz')
A = np.load(ROOT + '/auriga/out/gas_disc_evolution_au18.npz')

SIMS = [
    dict(tag='gastro', name='gastro Clumpy+merger',
         crit=r'$T<1.5\times10^4$ K, $n_H>1$ cm$^{-3}$',
         t=G['time'], rh=G['rhalf_sf'], r90=G['r90_sf'], m=G['m_sf'],
         rstar=G['rhalf_star'], colour='#2166ac', tmax=10.,
         events=[(1.6, '1st peri'), (2.5, '2nd peri'), (3.2, 'disrupted')],
         tref=3.2),
    dict(tag='au18', name='Auriga halo 18',
         crit=r'$n_H>0.13$ cm$^{-3}$ (SFR $>0$)',
         t=A['time'], rh=A['rhalf_sf'], r90=A['r90_sf'], m=A['m_sf'],
         rstar=A['rhalf_star'], colour='#b2182b', tmax=14.,
         events=[(3.25, 'first apo'), (5.0, 'plunge'), (5.4, 'coalescence')],
         tref=5.4),
]

fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))

# --- top row: each galaxy on its own axes ------------------------------------
for ax, S in zip(axes[0], SIMS):
    ax.plot(S['t'], S['rh'], 'o-', color=S['colour'], lw=2.3, ms=4,
            label='star-forming gas, $R_{1/2}$')
    ax.plot(S['t'], S['r90'], 's--', color=S['colour'], lw=1.2, ms=3, alpha=.55,
            label='star-forming gas, $R_{90}$')
    ax.plot(S['t'], S['rstar'], 'o-', color='.35', lw=1.6, ms=3, label='stars, $R_{1/2}$')
    for x, lab in S['events']:
        ax.axvline(x, color='k', lw=1.1, ls='--')
    ymax = np.nanmax(S['r90']) * 1.16
    for x, lab in S['events']:
        ax.text(x - .10, .015 * ymax, lab, rotation=90, ha='right', va='bottom', fontsize=8.5)
    ax.set(xlim=(0, S['tmax']), ylim=(0, ymax), xlabel='cosmic time [Gyr]',
           ylabel='half-mass radius [kpc]')
    ax.set_title(f"{S['name']}\n{S['crit']}", fontsize=11)
    ax.legend(fontsize=8.5, loc='upper left', framealpha=.95)

# --- bottom left: aligned and normalised -------------------------------------
ax = axes[1, 0]
for S in SIMS:
    dt = S['t'] - S['tref']
    ref = np.nanmean(S['rh'][(dt > -2.0) & (dt < -1.0)])
    ax.plot(dt, S['rh'] / ref, 'o-', color=S['colour'], lw=2.3, ms=4,
            label=f"{S['name']}  ($R_{{\\rm ref}}={ref:.2f}$ kpc)")
    S['ref'] = ref
ax.axvline(0, color='k', lw=1.2, ls='--')
ax.axhline(1, color='.6', lw=.8, ls=':')
ax.set(xlim=(-3, 6), xlabel='time since the merger ends [Gyr]',
       ylabel=r'$R_{1/2}$ / pre-merger $R_{1/2}$')
ax.set_title('Aligned on the final passage, each normalised to its own pre-merger size',
             fontsize=10.5)
ax.legend(fontsize=9)

# --- bottom right: the star-forming gas reservoir -----------------------------
ax = axes[1, 1]
for S in SIMS:
    dt = S['t'] - S['tref']
    ref = np.nanmean(S['m'][(dt > -2.0) & (dt < -1.0)])
    ax.plot(dt, S['m'] / ref, 'o-', color=S['colour'], lw=2.3, ms=4,
            label=f"{S['name']}  ($M_{{\\rm ref}}={ref/1e9:.2f}\\times10^9\\,M_\\odot$)")
ax.axvline(0, color='k', lw=1.2, ls='--')
ax.axhline(1, color='.6', lw=.8, ls=':')
ax.set(xlim=(-3, 6), xlabel='time since the merger ends [Gyr]',
       ylabel=r'$M_{\rm SF\,gas}$ / pre-merger value')
ax.set_title('Star-forming gas mass: does the disc shrink at fixed gas mass?', fontsize=10.5)
ax.legend(fontsize=9)

fig.suptitle('Star-forming gas disc size through the merger, each simulation on its own '
             'star-formation criterion', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, .94])
out = OUT + '/gas_disc_vs_merger.png'
fig.savefig(out, dpi=150)

# ------------------------------------------------------------------- numbers --
for S in SIMS:
    t, rh = S['t'], S['rh']
    dt = t - S['tref']
    pre = (dt > -2.0) & (dt < -1.0)
    win = (dt > -2.0) & (dt < 0.5)
    i = np.nanargmin(np.where(win, rh, np.inf))
    post = (dt > 1.0) & (dt < 3.0)
    print(f"\n{S['name']}  [{S['crit']}]")
    print(f"  pre-merger  R_1/2 = {np.nanmean(rh[pre]):5.2f} kpc  (mean over dt = -2 to -1 Gyr)")
    print(f"  minimum     R_1/2 = {rh[i]:5.2f} kpc at t = {t[i]:5.2f} Gyr "
          f"(dt = {dt[i]:+.2f})  -> contraction {100*(1-rh[i]/np.nanmean(rh[pre])):.0f}%")
    print(f"  1-3 Gyr on  R_1/2 = {np.nanmean(rh[post]):5.2f} kpc "
          f"({np.nanmean(rh[post])/np.nanmean(rh[pre]):.2f} x pre-merger)")
    print(f"  SF gas mass: pre {np.nanmean(S['m'][pre])/1e9:5.2f}e9 -> "
          f"at minimum {S['m'][i]/1e9:5.2f}e9 "
          f"({S['m'][i]/np.nanmean(S['m'][pre]):.2f} x)")
print('\nsaved', out)
