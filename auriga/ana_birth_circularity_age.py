"""Birth circularity of Au18 in-situ stars against cosmic time, and the SFR split by it.

eps = L_z/L_circ(E) measured in the first stored snapshot at or after each star
formed (<= 0.15 Gyr after birth), against an axisymmetric AGAMA CylSpline fitted
to the particle distribution of that epoch, in that epoch's own disc frame.  See
prep_potentials_agama.py and prep_birth_orbits_agama.py.

The classification uses circularity AND vertical extent, because circularity alone
cannot tell a halo orbit from a BAR orbit.  Bar orbits are planar and elongated with low
L_z for their energy, so they fall below any circularity cut; diag_bar_orientation.py
finds the inner low-eps newborns aligned with the bar to 0.9 deg at t = 6.2 Gyr and
2.2 deg at t = 9.4 Gyr (axis ratios b/a = 0.45 and 0.36).  Requiring a halo-born
star to be off-plane as well as slowly rotating removes them without a radius cut:

  disc-born   eps > 0.8  OR  z_max < ZCUT      (rotation-supported or planar)
  halo-born   eps <= 0.8 AND z_max >= ZCUT     (slowly rotating AND vertically extended)

MODE selects what stands in for "off-plane": 'zmax' uses the vertical excursion
derived from the birth vertical action (prep_zmax.py), which is phase-independent;
'epsz' uses the instantaneous |z| at birth, which is not -- it depends on where in
its orbit the snapshot caught the star; 'eps' drops the second condition entirely
and lets the bar into the halo-born class.

The z_max here underestimates the true value (by ~8 per cent for circular orbits
and up to ~40 per cent for eccentric ones -- see prep_zmax.py), so the 2 kpc cut
acts as a somewhat stricter threshold than its label suggests.  It is monotonic in
J_z, so it ranks orbits correctly, which is all a classifier needs.

The two are complements, so every star is classified exactly once.  Note this makes
disc-born the permissive class: a star has to fail both tests to count as halo-born.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import config_au18 as C

os.makedirs(C.FIG_DIR, exist_ok=True)
CUT, DT = 0.8, 0.15
# 'eps'  -- circularity alone, which lets the stellar bar into the halo-born class
# 'epsz' -- circularity AND height, which keeps the bar on the disc side
MODE = sys.argv[1] if len(sys.argv) > 1 else 'zmax'
ZCUT = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
SUF = {'eps': '_epsonly', 'epsz': '_epsz', 'zmax': ''}[MODE]
T_PERI, T_COAL_LO, T_COAL_HI = 5.0, 5.3, 5.6
T_SPIN = float(np.load(C.OUT_DIR + '/insitu_spinup.npz')['t_spin'])
EPOCHS = [('before the merger  (3.5-4.7 Gyr)', 3.5, 4.7, '#7b3294'),
          ('during the merger  (4.9-5.7 Gyr)', 4.9, 5.7, '#e66101'),
          ('after the merger   (6.6-8.0 Gyr)', 6.6, 8.0, '#018571')]
cD, cH, cM, cS = '#2166ac', '#b2182b', 'goldenrod', '#00857a'

a = np.load(C.OUT_DIR + '/birth_orbits_actions.npz')
zx = np.load(C.OUT_DIR + '/birth_orbits_zmax.npz')
q = np.load(C.OUT_DIR + '/insitu_imass.npz')
assert np.array_equal(a['ids'], q['ids']) and np.array_equal(a['ids'], zx['ids'])
tf, eb, mi = a['tform'], a['eps_birth'], q['imass']
zb = np.abs(a['z_birth']) if MODE == 'epsz' else zx['zmax_birth']
g = np.isfinite(eb) & np.isfinite(mi) & (np.isfinite(zb) | (MODE == 'eps'))
tf, eb, mi, zb = tf[g], eb[g], mi[g], zb[g]
if MODE == 'eps':
    disc = eb > CUT
    LAB_D, LAB_H = f'$\\epsilon>{CUT}$', f'$\\epsilon\\leq{CUT}$'
    SUB = 'circularity only'
else:
    Z = '|z|' if MODE == 'epsz' else 'z_{max}'
    disc = (eb > CUT) | (zb < ZCUT)
    LAB_D = f'$\\epsilon>{CUT}$ or ${Z}<{ZCUT:g}$ kpc'
    LAB_H = f'$\\epsilon\\leq{CUT}$ and ${Z}\\geq{ZCUT:g}$ kpc'
    SUB = ('halo-born = slowly rotating AND off-plane' if MODE == 'epsz'
           else 'halo-born = slowly rotating AND vertically extended ($z_{max}$)')
halo = ~disc
TMIN = np.floor(tf.min() * 10) / 10
print(f'{g.sum():,} stars; disc-born {disc.sum():,} ({100 * disc.mean():.1f}%), '
      f'halo-born {halo.sum():,} ({100 * halo.mean():.1f}%)')

# Cross-check the project's spin-up time against the birth circularity itself.
tb = np.arange(TMIN, C.T0_GYR, .2)
med = np.array([np.median(eb[(tf >= x) & (tf < x + .2)]) if ((tf >= x) & (tf < x + .2)).sum() > 200
                else np.nan for x in tb])
above = np.flatnonzero(np.convolve((med > .7).astype(float), np.ones(3) / 3, 'same') > .99)
print(f'spin-up: project value (z=0 eps) {T_SPIN:.2f} Gyr; '
      f'from birth eps, median crosses 0.7 at {tb[above[0]] + .1 if len(above) else np.nan:.2f} Gyr')

fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.4))

# ---- (a) column-normalised: the shape of the eps distribution at every epoch ---
ax = axes[0]
tbins = np.arange(TMIN, C.T0_GYR + DT, DT)
ebins = np.linspace(-1, 1, 81)
H, _, _ = np.histogram2d(tf, eb, bins=[tbins, ebins], weights=mi)
col = H.sum(1)
Hn = np.divide(H, col[:, None], out=np.full_like(H, np.nan), where=col[:, None] > 0)
# Linear colour scale, but saturated well below the peak: the eps ~ 0.97 bin holds
# up to 70 per cent of a column's mass, and scaling to that leaves everything else
# indistinguishable from white.
VMAX = float(np.nanpercentile(Hn, 98))
pc = ax.pcolormesh(tbins, ebins, Hn.T, cmap='Oranges', vmin=0, vmax=VMAX)
print(f'panel (a) colour scale: linear, 0 to {VMAX:.3f} (98th pct); '
      f'peak bin {np.nanmax(Hn):.3f}')
ax.axhline(CUT, color='.25', ls='--', lw=1.3)
ax.axhline(0, color='.45', lw=.6, alpha=.7)
ax.axvspan(T_COAL_LO, T_COAL_HI, color=cM, alpha=.45, lw=0)
ax.axvline(T_PERI, color=cM, ls='--', lw=1.5)
ax.set(xlim=(TMIN, C.T0_GYR), ylim=(-1, 1), xlabel='cosmic time [Gyr]',
       ylabel='$\\epsilon$ at birth',
       title='(a) Birth circularity against time (column-normalised)')
fig.colorbar(pc, ax=ax, pad=.01, extend='max',
             label='fraction of the mass formed at that time')

# ---- (b) the same information as three distributions --------------------------
ax = axes[1]
for lab, lo, hi, c in EPOCHS:
    m = (tf >= lo) & (tf < hi)
    ax.hist(eb[m], bins=np.linspace(-1, 1, 81), weights=mi[m], density=True,
            histtype='step', lw=2.2, color=c,
            label=f'{lab}\n    {100 * mi[m & halo].sum() / mi[m].sum():.1f}% halo-born')
ax.axvline(CUT, color='.3', ls='--', lw=1.4)
ax.set(xlim=(-.2, 1), xlabel='$\\epsilon$ at birth', ylabel='normalised density (mass-weighted)',
       title='(b) Birth circularity before, during and after')
ax.legend(fontsize=8.5, loc='upper left')

# ---- (c) the two channels as star-formation histories, and their ratio ---------
ax = axes[2]
ctr = .5 * (tbins[:-1] + tbins[1:])
sd = np.histogram(tf[disc], bins=tbins, weights=mi[disc])[0] / (DT * 1e9)
sh = np.histogram(tf[halo], bins=tbins, weights=mi[halo])[0] / (DT * 1e9)
ax.axvline(T_SPIN, color=cS, ls=':', lw=2.2, label=f'disc spin-up ({T_SPIN:.1f} Gyr)')
ax.axvline(T_PERI, color=cM, ls='--', lw=2, label=f'GS/E pericentre ({T_PERI:.1f} Gyr)')
ax.step(ctr, sd, where='mid', color=cD, lw=2,
        label=f'disc-born ({LAB_D})')
ax.fill_between(ctr, 0, sd, step='mid', color=cD, alpha=.20, lw=0)
ax.step(ctr, sh, where='mid', color=cH, lw=2,
        label=f'halo-born ({LAB_H})')
ax.fill_between(ctr, 0, sh, step='mid', color=cH, alpha=.30, lw=0)
ax.set(xlim=(TMIN, C.T0_GYR), ylim=(0, 1.08 * max(sd.max(), sh.max())),
       xlabel='cosmic time [Gyr]', ylabel='SFR [M$_\\odot$ yr$^{-1}$]',
       title='(c) The two channels, and their ratio')
ax.legend(fontsize=8.5, loc='upper right')
axr = ax.twinx()
solid = (sd + sh) > 2.                       # drop the poorly-sampled earliest bins
rat = np.divide(sh, sd, out=np.full_like(sh, np.nan), where=(sd > 0) & solid)
axr.plot(ctr, rat, color='k', lw=1.7, ls='-.')
axr.set_ylabel('halo-born / disc-born SFR (dash-dot)', fontsize=9.5)
axr.set_ylim(0, 1.5)          # the pre-spin-up ratio runs off the top; the merger is the feature

pk = np.nanargmax(np.where(solid, sh, np.nan))
print(f'\nhalo-born SFR peaks at t = {ctr[pk]:.2f} Gyr ({sh[pk]:.2f} Msun/yr); '
      f'ratio there {rat[pk]:.2f}')
for lab, lo, hi, _ in EPOCHS:
    m = (tf >= lo) & (tf < hi)
    print(f'  {lab:34s} halo-born {100 * mi[m & halo].sum() / mi[m].sum():5.1f}%   '
          f'halo/disc SFR ratio = {mi[m & halo].sum() / mi[m & disc].sum():.3f}')

fig.suptitle(f'Au18: the orbits stars are born on, through the GS/E merger  ({SUB})', y=.99)
fig.tight_layout(rect=[0, 0, 1, .94])
out = C.FIG_DIR + f'/au18_birth_circularity_age{SUF}.png'
fig.savefig(out, dpi=150)
print('\nsaved', out)
