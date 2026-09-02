"""Is eps = 0.5 a real boundary, and which stars change sides between birth and now?

Three things, for the stars formed during the GS/E burst (t = 4.9-5.7 Gyr):

1. The birth-circularity distribution is UNIMODAL -- there is no density minimum
   anywhere, at birth or at z=0.  eps = 0.5 is therefore a convention, not a
   natural boundary, and the absolute "halo-born fraction" is meaningless unless
   the cut is quoted with it.  Panel (b) shows what does survive the choice -- the
   burst-to-quiet RATIO, which with the AGAMA circularity is flat at 2.3-2.4 for
   any cut between 0.2 and 0.8.
2. eps at birth against eps at z=0, so the two-way traffic is visible rather than
   assumed: stars born on the disc that were later heated off it, and stars born
   hot that later settled onto disc orbits.
3. The v_R-v_phi plane at birth and at z=0 for the same cohort.

eps at birth and at z=0 both come from prep_birth_orbits_agama.py: L_z/L_circ(E)
against an axisymmetric CylSpline fitted to the particle distribution, each epoch
in its own disc frame.  The 95th-percentile-envelope estimator this script used
first put 11.6 per cent of stars above eps = 1 and concentrated its halo-to-disc
transitions at R_birth ~ 2 kpc, i.e. it was failing in the centre; with the real
potential that anomaly is gone (4.08 vs 3.98 kpc).  Birth velocities still come
from prep_birth_orbits.py, which measured them in the same contemporary disc frame
-- they depend on the alignment, not on the potential.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import config_au18 as C

os.makedirs(C.FIG_DIR, exist_ok=True)

CUT = 0.5
BURST = (4.9, 5.7)
EPOCHS = [('quiet before 3.5-4.7', 3.5, 4.7, '#7b3294'),
          ('burst 4.9-5.7', BURST[0], BURST[1], '#e66101'),
          ('quiet after 6.6-8.0', 6.6, 8.0, '#018571')]
cD, cH = '#2166ac', '#b2182b'

b = np.load(C.OUT_DIR + '/birth_orbits_agama.npz')
vel = np.load(C.OUT_DIR + '/birth_orbits.npz')          # birth velocities only
assert np.array_equal(b['ids'], vel['ids'])
q = np.load(C.OUT_DIR + '/insitu_imass.npz')
cat = np.load(C.OUT_DIR + '/z0_insitu_catalog.npz')
o = np.argsort(cat['ids']); sid = cat['ids'][o]
p = np.searchsorted(sid, b['ids'])
ok = (p < len(sid)) & (sid[np.minimum(p, len(sid) - 1)] == b['ids'])
ix = o[p[ok]]

tf = b['tform'][ok]; eb = b['eps_birth'][ok]; mi = q['imass'][ok]
vRb, vpb = vel['vR_birth'][ok], vel['vphi_birth'][ok]
ez = b['eps_z0'][ok]; vRz, vpz = cat['vR'][ix], cat['vphi'][ix]
g = np.isfinite(eb) & np.isfinite(ez) & np.isfinite(mi)
tf, eb, ez, mi, vRb, vpb, vRz, vpz = (a[g] for a in (tf, eb, ez, mi, vRb, vpb, vRz, vpz))
print(f'joined {len(tf):,} in-situ stars with both birth and z=0 circularity')

bur = (tf >= BURST[0]) & (tf <= BURST[1])
db, hb = eb >= CUT, eb < CUT
dz, hz = ez >= CUT, ez < CUT

print(f'\nburst cohort t = {BURST[0]}-{BURST[1]} Gyr: N = {bur.sum():,}, '
      f'M_init = {mi[bur].sum():.3e} Msun')
print(f'\ntransition matrix at eps = {CUT} (per cent of the burst cohort)')
print(f'{"":26s} {"N":>9s} {"M_init":>10s} {"% cohort":>9s} {"% of its birth class":>21s}')
cells = [('born disc -> still disc', db & dz), ('born disc -> now halo  ', db & hz),
         ('born halo -> now disc  ', hb & dz), ('born halo -> still halo', hb & hz)]
for lab, m in cells:
    parent = db if lab.startswith('born disc') else hb
    mm = m & bur
    print(f'{lab:26s} {mm.sum():9,} {mi[mm].sum():10.3e} '
          f'{100 * mm.sum() / bur.sum():8.1f}% {100 * mm.sum() / (parent & bur).sum():20.1f}%')
print(f'net flow disc->halo minus halo->disc: '
      f'{(db & hz & bur).sum() - (hb & dz & bur).sum():+,} stars')

print(f'\nsensitivity of the answer to the cut (burst cohort)')
print(f'{"cut":>5s} {"halo-born":>10s} {"stays disc":>11s} {"disc->halo":>11s} '
      f'{"halo->disc":>11s} {"burst/quiet":>12s}')
QB = (tf >= 3.5) & (tf < 4.7)
for c in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
    hbc, hzc = eb < c, ez < c
    fb = mi[bur & hbc].sum() / mi[bur].sum()
    fq = mi[QB & hbc].sum() / mi[QB].sum()
    print(f'{c:5.1f} {100 * fb:9.1f}% {100 * (bur & ~hbc & ~hzc).sum() / (bur & ~hbc).sum():10.1f}% '
          f'{100 * (bur & ~hbc & hzc).sum() / (bur & ~hbc).sum():10.1f}% '
          f'{100 * (bur & hbc & ~hzc).sum() / (bur & hbc).sum():10.1f}% {fb / fq:12.2f}')

# ------------------------------------------------------------------ figure --
fig, axes = plt.subplots(2, 3, figsize=(18.2, 9.6))
ebins = np.linspace(-1.2, 1.4, 110)

# (a) no dip anywhere: the cut is a convention
ax = axes[0, 0]
for lab, lo, hi, c in EPOCHS:
    m = (tf >= lo) & (tf < hi)
    ax.hist(eb[m], bins=ebins, weights=mi[m], density=True, histtype='step', lw=2,
            color=c, label=f'{lab} (birth)')
m = bur
ax.hist(ez[m], bins=ebins, weights=mi[m], density=True, histtype='stepfilled',
        color='.5', alpha=.25, lw=0, label='burst, at z=0')
for c in (0.3, 0.5, 0.7):
    ax.axvline(c, color='.3', ls=':' if c != CUT else '--', lw=1.4)
ax.text(CUT, ax.get_ylim()[1] * .97, ' $\\epsilon=0.5$', fontsize=8.5, va='top')
ax.set(xlabel='$\\epsilon$', ylabel='normalised density (mass-weighted)', yscale='log',
       ylim=(2e-3, 6), title='(a) No density minimum: the cut is a convention')
ax.legend(fontsize=8, loc='upper left')

# (b) what survives the choice of cut
ax = axes[0, 1]
cuts = np.linspace(.15, .85, 36)
for lab, lo, hi, c in EPOCHS:
    m = (tf >= lo) & (tf < hi)
    ax.plot(cuts, [100 * mi[m & (eb < x)].sum() / mi[m].sum() for x in cuts],
            color=c, lw=2, label=lab)
ax.axvline(CUT, color='.3', ls='--', lw=1.4)
ax.set(xlabel='circularity cut', ylabel='per cent classified halo-born',
       title='(b) The fraction depends on the cut; the ratio does not')
ax.legend(fontsize=8.5, loc='upper left')
axr = ax.twinx()
fb = np.array([mi[bur & (eb < x)].sum() / mi[bur].sum() for x in cuts])
fq = np.array([mi[QB & (eb < x)].sum() / mi[QB].sum() for x in cuts])
axr.plot(cuts, fb / fq, color='k', lw=1.6, ls='-.')
axr.axhline(1, color='k', lw=.6, alpha=.4)
axr.set_ylabel('burst / quiet-before ratio (dash-dot)', fontsize=9)
axr.set_ylim(0, 2.6)

# (c) the two-way traffic
ax = axes[0, 2]
h = ax.hist2d(eb[bur], ez[bur], bins=[np.linspace(-1.2, 1.4, 120)] * 2,
              cmin=1, cmap='viridis', norm=LogNorm())
ax.axvline(CUT, color='w', lw=1.3, ls='--'); ax.axhline(CUT, color='w', lw=1.3, ls='--')
ax.plot([-1.2, 1.4], [-1.2, 1.4], color='w', lw=.8, alpha=.5)
# The left half of the panel is mostly empty bins, i.e. white, so the labels
# there need their own dark background rather than relying on the colormap.
for (lab, m), (xx, yy) in zip(cells, [(1.05, 1.15), (1.05, -1.0), (-.85, 1.15), (-.85, -1.0)]):
    ax.text(xx, yy, f'{100 * (m & bur).sum() / bur.sum():.1f}%', color='w', fontsize=11,
            ha='center', va='center', fontweight='bold',
            bbox=dict(fc='0.15', ec='none', alpha=.8, pad=2.5))
ax.set(xlabel='$\\epsilon$ at birth', ylabel='$\\epsilon$ at z=0',
       title='(c) Burst cohort: birth orbit vs orbit today')
fig.colorbar(h[3], ax=ax, pad=.01, label='stars per bin')

# (d, e) the plane the observers actually see
for a, xv, yv, ttl in [(axes[1, 0], vRb[bur], vpb[bur], '(d) Burst cohort at birth'),
                       (axes[1, 1], vRz[bur], vpz[bur], '(e) The same stars at z=0')]:
    hh = a.hist2d(xv, yv, bins=140, range=[[-350, 350], [-300, 400]], cmin=1,
                  cmap='magma', norm=LogNorm())
    a.axhline(0, color='.7', lw=.6); a.axvline(0, color='.7', lw=.6)
    a.set(xlabel='$v_R$ [km s$^{-1}$]', ylabel='$v_\\phi$ [km s$^{-1}$]', title=ttl)
    a.text(.03, .97, f'$\\langle v_\\phi\\rangle$ = {np.mean(yv):.0f}\n'
           f'$\\sigma_R$ = {np.std(xv):.0f}\n$\\sigma_\\phi$ = {np.std(yv):.0f}',
           transform=a.transAxes, va='top', fontsize=9,
           bbox=dict(fc='white', alpha=.75, ec='none'))
    fig.colorbar(hh[3], ax=a, pad=.01)

# (f) where each birth class ended up
ax = axes[1, 2]
for i, (parent, plab, pc) in enumerate([(db, 'born on\ndisc orbits', cD),
                                        (hb, 'born on\nhalo orbits', cH)]):
    n = (parent & bur).sum()
    stay = (parent & bur & (dz if i == 0 else hz)).sum()
    ax.barh(i, 100 * stay / n, color=pc, alpha=.75, height=.55,
            label='stays in its birth class' if i == 0 else None)
    ax.barh(i, 100 * (n - stay) / n, left=100 * stay / n, color=pc, alpha=.28, height=.55,
            hatch='//', label='changes class by z=0' if i == 0 else None)
    ax.text(2, i + .34, f'{plab.splitlines()[0]} {plab.splitlines()[1]}   N = {n:,}',
            fontsize=9, va='center')
    ax.text(100 * stay / n / 2, i, f'{100 * stay / n:.0f}%', ha='center', va='center',
            color='w', fontsize=12, fontweight='bold')
    ax.text(100 * (1 + stay / n) / 2, i, f'{100 * (n - stay) / n:.0f}%', ha='center',
            va='center', fontsize=12)
ax.set(yticks=[], xlim=(0, 100), xlabel='per cent of that birth class',
       title=f'(f) Where the burst stars ended up ($\\epsilon$ cut {CUT})')
ax.legend(fontsize=8.5, loc='lower right')
ax.invert_yaxis()

fig.suptitle('Au18 GS/E burst cohort: how the birth/present orbit split actually behaves', y=.985)
fig.tight_layout(rect=[0, 0, 1, .95])
out = C.FIG_DIR + '/au18_birth_orbit_transitions.png'
fig.savefig(out, dpi=140)
print('\nsaved', out)
