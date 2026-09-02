"""Birth and present orbits of Au18 in-situ stars, from the AGAMA CylSpline potentials.

Supersedes the eps columns in ana_birth_orbit_sfh.py and ana_birth_orbit_transitions.py,
which used jz divided by the 95th-percentile prograde envelope in each energy bin.
That estimator is normalised by the star distribution rather than by the potential:
it put 11.6 per cent of stars above eps = 1 at birth, it drifted between a
disordered z = 2 galaxy and the settled z = 0 disc, and it failed hardest in the
central few kpc -- which is where it invented most of the halo-to-disc traffic.

Here eps = L_z/L_circ(E) with L_circ from an axisymmetric CylSpline fitted to the
actual particle distribution (prep_potentials_agama.py), each snapshot measured in
its own disc frame, nearest potential in time.  The maximum angle between a
snapshot's disc axis and its assigned potential's axis is 14.8 degrees, and 0 in
the merger window where the potentials are refined to every snapshot.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import config_au18 as C

os.makedirs(C.FIG_DIR, exist_ok=True)
CUT, DT = 0.5, 0.15
T_PERI, T_COAL_LO, T_COAL_HI = 5.0, 5.3, 5.6
EPOCHS = [('quiet before 3.5-4.7', 3.5, 4.7, '#7b3294'),
          ('burst 4.9-5.7', 4.9, 5.7, '#e66101'),
          ('quiet after 6.6-8.0', 6.6, 8.0, '#018571')]
cD, cH, cM = '#2166ac', '#b2182b', 'goldenrod'

a = np.load(C.OUT_DIR + '/birth_orbits_agama.npz')
old = np.load(C.OUT_DIR + '/birth_orbits.npz')
q = np.load(C.OUT_DIR + '/insitu_imass.npz')
assert np.array_equal(a['ids'], q['ids']) and np.array_equal(a['ids'], old['ids'])
tf, eb, ez, mi = a['tform'], a['eps_birth'], a['eps_z0'], q['imass']
g = np.isfinite(eb) & np.isfinite(ez) & np.isfinite(mi)
tf, eb, ez, mi = tf[g], eb[g], ez[g], mi[g]
hb, hz = eb < CUT, ez < CUT
QB, BU, QA = [(tf >= lo) & (tf < hi) for _, lo, hi, _ in EPOCHS]

print(f'{g.sum():,} stars with eps at birth and at z=0')
print(f'{"estimator":16s} {"eps>1 birth":>12s} {"eps>1 z=0":>11s} {"halo->disc":>11s} '
      f'{"disc->halo":>11s} {"medR_birth h->d":>16s}')
for lab, b_, z_ in [('envelope', old['eps_z0'][g] * 0 + old['eps_birth'][g], old['eps_z0'][g]),
                    ('spherical', old['eps_sph_birth'][g], old['eps_sph_z0'][g]),
                    ('AGAMA CylSpline', eb, ez)]:
    h1, h2 = b_ < CUT, z_ < CUT
    print(f'{lab:16s} {100 * np.nanmean(b_ > 1):11.2f}% {100 * np.nanmean(z_ > 1):10.2f}% '
          f'{100 * (BU & h1 & ~h2).sum() / (BU & h1).sum():10.1f}% '
          f'{100 * (BU & ~h1 & h2).sum() / (BU & ~h1).sum():10.1f}% '
          f'{np.median(a["R_birth"][g][BU & h1 & ~h2]):15.2f}')

print(f'\ntransition matrix, burst cohort (N = {BU.sum():,}), eps cut {CUT}')
cells = [('born disc -> still disc', ~hb & ~hz), ('born disc -> now halo', ~hb & hz),
         ('born halo -> now disc', hb & ~hz), ('born halo -> still halo', hb & hz)]
for lab, m in cells:
    par = (BU & ~hb) if lab.startswith('born disc') else (BU & hb)
    print(f'  {lab:26s} N={(m & BU).sum():>7,}  {100 * (m & BU).sum() / BU.sum():5.1f}% of cohort'
          f'  {100 * (m & BU).sum() / par.sum():5.1f}% of class')
print(f'  net flow disc->halo minus halo->disc: {(~hb & hz & BU).sum() - (hb & ~hz & BU).sum():+,}'
      f'  ratio {(~hb & hz & BU).sum() / (hb & ~hz & BU).sum():.1f} : 1')

# ------------------------------------------------------------------ figure --
fig, axes = plt.subplots(2, 3, figsize=(18.2, 9.6))
ebins = np.linspace(-1.3, 1.3, 115)

ax = axes[0, 0]
for lab, arr, c, ls in [('95th-pct envelope', old['eps_z0'][g], '#999999', '-'),
                        ('spherical $\\Phi(r)$', old['eps_sph_z0'][g], '#66a61e', '-'),
                        ('AGAMA CylSpline', ez, 'k', '-')]:
    ax.hist(arr[np.isfinite(arr)], bins=ebins, weights=mi[np.isfinite(arr)], density=True,
            histtype='step', lw=2, color=c, ls=ls,
            label=f'{lab}  ({100 * np.nanmean(arr > 1):.1f}% above 1)')
ax.axvline(1, color='r', lw=1.2, ls='--')
ax.set(xlabel='$\\epsilon$ at z=0', ylabel='normalised density', yscale='log', ylim=(1e-3, 8),
       title='(a) Why the estimator was changed')
ax.legend(fontsize=8, loc='upper left')

ax = axes[0, 1]
for lab, lo, hi, c in EPOCHS:
    m = (tf >= lo) & (tf < hi)
    ax.hist(eb[m], bins=ebins, weights=mi[m], density=True, histtype='step', lw=2,
            color=c, label=lab)
ax.axvline(CUT, color='.3', ls='--', lw=1.3)
ax.set(xlabel='$\\epsilon$ at birth', ylabel='normalised density', yscale='log', ylim=(1e-3, 8),
       title='(b) Still unimodal: no natural boundary')
ax.legend(fontsize=8.5, loc='upper left')

ax = axes[0, 2]
cuts = np.linspace(.15, .85, 36)
for lab, lo, hi, c in EPOCHS:
    m = (tf >= lo) & (tf < hi)
    ax.plot(cuts, [100 * mi[m & (eb < x)].sum() / mi[m].sum() for x in cuts], color=c, lw=2,
            label=lab)
ax.axvline(CUT, color='.3', ls='--', lw=1.3)
ax.set(xlabel='circularity cut', ylabel='per cent halo-born',
       title='(c) The ratio is now flat in the cut')
ax.legend(fontsize=8.5, loc='upper left')
axr = ax.twinx()
fb = np.array([mi[BU & (eb < x)].sum() / mi[BU].sum() for x in cuts])
fq = np.array([mi[QB & (eb < x)].sum() / mi[QB].sum() for x in cuts])
axr.plot(cuts, fb / fq, color='k', lw=1.6, ls='-.')
axr.axhline(1, color='k', lw=.6, alpha=.4)
axr.set(ylabel='burst / quiet ratio (dash-dot)', ylim=(0, 3.2))

ax = axes[1, 0]
h = ax.hist2d(eb[BU], ez[BU], bins=[np.linspace(-1.3, 1.3, 120)] * 2, cmin=1,
              cmap='viridis', norm=LogNorm())
ax.axvline(CUT, color='w', lw=1.3, ls='--'); ax.axhline(CUT, color='w', lw=1.3, ls='--')
ax.plot([-1.3, 1.3], [-1.3, 1.3], color='w', lw=.8, alpha=.5)
for (lab, m), (xx, yy) in zip(cells, [(1.05, 1.15), (1.05, -1.1), (-.95, 1.15), (-.95, -1.1)]):
    ax.text(xx, yy, f'{100 * (m & BU).sum() / BU.sum():.1f}%', color='w', fontsize=11,
            ha='center', va='center', fontweight='bold',
            bbox=dict(fc='0.15', ec='none', alpha=.8, pad=2.5))
ax.set(xlabel='$\\epsilon$ at birth', ylabel='$\\epsilon$ at z=0',
       title='(d) Burst cohort: heating outruns settling 5:1')
fig.colorbar(h[3], ax=ax, pad=.01)

ax = axes[1, 1]
bins = np.arange(np.floor(tf.min() * 10) / 10, C.T0_GYR + DT, DT)
ctr = .5 * (bins[:-1] + bins[1:])
sa = np.histogram(tf, bins=bins, weights=mi)[0] / (DT * 1e9)
sh = np.histogram(tf[hb], bins=bins, weights=mi[hb])[0] / (DT * 1e9)
ax.axvspan(T_COAL_LO, T_COAL_HI, color=cM, alpha=.45, lw=0, label='GS/E coalescence')
ax.axvline(T_PERI, color=cM, ls='--', lw=1.5, label='pericentre plunge')
ax.fill_between(ctr, 0, sh, step='mid', color=cH, alpha=.55, lw=0, label='born on halo orbits')
ax.fill_between(ctr, sh, sa, step='mid', color=cD, alpha=.45, lw=0, label='born on disc orbits')
ax.step(ctr, sa, where='mid', color='k', lw=1.3)
ax.set(xlim=(tf.min(), C.T0_GYR), ylim=(0, 1.1 * sa.max()), xlabel='cosmic time [Gyr]',
       ylabel='SFR [M$_\\odot$ yr$^{-1}$]', title='(e) SFH split by AGAMA birth orbit')
ax.legend(fontsize=8.5, loc='upper right')

ax = axes[1, 2]
tilt = np.degrees(np.arccos(np.clip(a['axis_now'] @ a['axis_z0'], -1, 1)))
ok = np.isfinite(tilt)
ax.plot(a['t_snap'][ok], tilt[ok], color='k', lw=2, marker='o', ms=3)
ax.axvspan(T_COAL_LO, T_COAL_HI, color=cM, alpha=.45, lw=0)
ax.axvline(T_PERI, color=cM, ls='--', lw=1.5)
ax.plot(a['t_snap'][ok], a['axis_gap'][ok], color='#66a61e', lw=1.4, ls='--',
        label='snapshot vs assigned potential')
ax.set(xlim=(tf.min(), C.T0_GYR), xlabel='cosmic time [Gyr]',
       ylabel='angle from the z=0 disc axis [deg]',
       title='(f) The disc reorients through the merger')
ax.legend(fontsize=8.5, loc='upper right')

fig.suptitle('Au18 birth orbits from AGAMA CylSpline potentials, each epoch in its own disc frame',
             y=.985)
fig.tight_layout(rect=[0, 0, 1, .95])
out = C.FIG_DIR + '/au18_birth_orbit_agama.png'
fig.savefig(out, dpi=140)
print('\nsaved', out)
