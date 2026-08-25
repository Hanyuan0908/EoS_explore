"""Star-forming gas disc size against time, for every simulation with the coverage
to show it, against the GS/E merger times reported in the literature.

Auriga halos are the GS/E analogues of Fattahi et al. (2019, MNRAS 484, 4471):
the ten that pass their beta > 0.8 and contribution > 50% cuts, of which
Au-5, 9, 10, 18 are called the extreme cases.  Their merger times were read off
that paper's Fig. 5 (bottom panel) from the PDF text coordinates; the axis fit is
exact and the extraction independently reproduces the list of ten.  As a check,
the paper puts Au-18's merger at 5.23 Gyr while our own dating of the same event
(auriga/PROGRESS.md, phase-space dispersion of the satellite) gave 5.3-5.6 Gyr.

Only three of the ten have snapshots that actually bracket their merger; the rest
either hold a single z=0 output or begin after the event.  gastro Clumpy+merger
is shown alongside for contrast -- a different code and an idealised setup, with
its own disruption time of 3.2 Gyr rather than a Fattahi value.

Disc definition is each simulation's own star-formation criterion:
  Auriga  the code's StarFormationRate flag, n_H > 0.13 cm^-3 (Grand et al. 2017)
  gastro  T < 1.5e4 K and n_H > 1 cm^-3 (Amarante et al. 2022)

Writes figures_sim/gas_disc_all_sims.png.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = ROOT + '/figures_sim'
os.makedirs(OUT, exist_ok=True)

# Fattahi et al. (2019) Fig. 5: merger time of the main contributor, in Gyr.
T_MERGE = {5: 7.26, 9: 3.54, 10: 7.41, 15: 8.52, 17: 3.38,
           18: 5.23, 22: 4.75, 24: 4.91, 26: 4.75, 27: 4.59}
EXTREME = {5, 9, 10, 18}
COLOUR = {5: '#1a9850', 18: '#b2182b', 26: '#762a83', 9: '#e08214', 27: '#4393c3'}

sims = []
for h in sorted(T_MERGE):
    p = f'{ROOT}/auriga/out/gas_disc_evolution_au{h}.npz'
    if not os.path.exists(p):
        continue
    d = np.load(p)
    t, rh = d['time'], d['rhalf_sf']
    good = np.isfinite(rh)
    tm = T_MERGE[h]
    # Does the run actually bracket its merger?  Require a pre-merger baseline.
    # Usable needs a pre-merger baseline AND enough cadence across the event.
    # Au-5 fails the second test: 9 snapshots in dt = -2.5..+1.5 with a 5-snapshot
    # (0.8 Gyr) gap straddling its merger, against 26 at full cadence for Au-18
    # and Au-26.  A contraction lasting ~0.5 Gyr cannot be resolved through that.
    near = good & (t > tm - 2.5) & (t < tm + 1.5)
    covers = good.sum() > 5 and t[good].min() < tm - 1.0
    well_sampled = covers and near.sum() >= 15
    sims.append(dict(h=h, t=t, rh=rh, m=d['m_sf'], tm=tm, covers=covers,
                     well=well_sampled, nnear=int(near.sum()),
                     colour=COLOUR.get(h, '.4'),
                     lab=f'Au-{h}' + ('*' if h in EXTREME else '')))
    print(f"Au-{h:<3d} t_merge={tm:5.2f}  snapshots {t.min():5.2f}-{t.max():5.2f} Gyr  "
          f"{near.sum():2d} snapshots across the merger  "
          f"{'USABLE' if well_sampled else ('too sparse - shown but not counted' if covers else 'STARTS AFTER THE MERGER - excluded')}")

gp = f'{ROOT}/gastro/out/gas_disc_evolution.npz'
gastro = None
if os.path.exists(gp):
    g = np.load(gp)
    gastro = dict(t=g['time'], rh=g['rhalf_sf'], m=g['m_sf'], tm=3.2,
                  colour='k', lab='gastro Clumpy+merger')

usable = [s for s in sims if s['covers']]
well = [s for s in sims if s['well']]
fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))

# --- (a) absolute, each halo with its own reported merger time ---------------
ax = axes[0]
for s in usable:
    ls = '-' if s['well'] else ':'
    tag = '' if s['well'] else ', sparse'
    ax.plot(s['t'], s['rh'], 'o' + ls, color=s['colour'], lw=2.1, ms=3.5,
            label=f"{s['lab']}  ($t_{{\\rm merge}}={s['tm']:.2f}$ Gyr{tag})")
    ax.axvline(s['tm'], color=s['colour'], lw=1.6, ls='--', alpha=.85)
ax.set(xlabel='cosmic time [Gyr]', ylabel=r'$R_{1/2}$ of the star-forming gas [kpc]',
       xlim=(0, 14))
ax.set_title('Auriga GS/E analogues: dashed = merger time from Fattahi et al. (2019)',
             fontsize=10.5)
ax.legend(fontsize=9, loc='upper left')

# --- (b) aligned on the reported merger, normalised --------------------------
ax = axes[1]
for s in usable + ([gastro] if gastro else []):
    dt = s['t'] - s['tm']
    ref = np.nanmean(s['rh'][(dt > -2.5) & (dt < -1.0)])
    if not np.isfinite(ref):
        continue
    ls = ':' if (s is gastro or not s.get('well', True)) else '-'
    ax.plot(dt, s['rh'] / ref, 'o' + ls, color=s['colour'], lw=2.1, ms=3.5,
            label=f"{s['lab']}  ($R_{{\\rm ref}}={ref:.1f}$ kpc)")
ax.axvline(0, color='k', lw=1.3, ls='--')
ax.axhline(1, color='.6', lw=.8, ls=':')
ax.set(xlim=(-3, 5), xlabel='time since the reported merger [Gyr]',
       ylabel=r'$R_{1/2}$ / pre-merger value')
ax.set_title('Aligned on each merger, normalised to its own pre-merger size', fontsize=10.5)
ax.legend(fontsize=9, loc='upper left')

fig.suptitle('Star-forming gas disc size versus the reported GS/E merger time '
             '(* = Fattahi et al. extreme analogue)', fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, .93])
out = OUT + '/gas_disc_all_sims.png'
fig.savefig(out, dpi=150)

# Windowed means, not a single-snapshot minimum: the early discs are noisy enough
# that picking the lowest point can land on an outlier.
print('\n%-24s %8s %8s %8s %9s' % ('', 'pre', 'merger', 'post', 'change'))
for s in usable + ([gastro] if gastro else []):
    dt = s['t'] - s['tm']
    pre, mer, post = (dt > -2.5) & (dt < -1.0), (dt > -.6) & (dt < .4), (dt > .8) & (dt < 2.)
    if not np.isfinite(s['rh'][pre]).any() or not np.isfinite(s['rh'][mer]).any():
        continue
    a, b, c = (np.nanmean(s['rh'][m]) for m in (pre, mer, post))
    flag = '' if s.get('well', True) else '   <- under-sampled, not reliable'
    print('%-24s %8.2f %8.2f %8.2f %8.0f%%%s' % (s['lab'], a, b, c, 100 * (b / a - 1), flag))
print('\nsaved', out)
