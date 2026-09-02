"""Was the GS/E starburst making disc stars or halo stars?

Splits the Au18 star-formation history by the orbit each star was BORN on, using
eps = L_z/L_circ(E) from the AGAMA CylSpline potentials (prep_birth_orbits_agama.py),
each snapshot measured in its own disc frame.  The split is at eps = 0.5: above it
a star is on a rotation-supported orbit, below it a pressure-supported one.

eps = 0.5 is a convention, not a boundary the data shows -- the distribution is
unimodal with no interior minimum -- so panels (c) and (d) show the whole eps
distribution.  What justifies still using a cut is that the burst-to-quiet RATIO
is flat in it (2.42, 2.37, 2.32 at cuts 0.3, 0.5, 0.7); see ana_birth_orbit_agama.py.

The SFR uses GFM_InitialMass, matched from the z=0 snapshot, so it is directly
comparable with ana_sfh_au18.py.  The in-situ sample is z0_insitu_catalog.npz
(1.98M stars), marginally smaller than the r < 0.15 R200 sample used there, so
the absolute SFR differs by a few per cent.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import config_au18 as C

os.makedirs(C.FIG_DIR, exist_ok=True)

DT = 0.15                                  # ~ the snapshot spacing
EPS_CUT = 0.5
T_APO, T_PERI = 3.25, 5.0
T_COAL_LO, T_COAL_HI = 5.3, 5.6
T_WIN = (4.99, 6.54)
cD, cH, cM = '#2166ac', '#b2182b', 'goldenrod'
# Quiet windows either side of the merger, and the burst itself.  The burst
# window is narrow on purpose: the halo-born excess in panel (b) lasts only
# ~0.5 Gyr, and averaging over the whole 4.99-6.54 Eos window dilutes it away.
EPOCHS = [('quiet before 3.5-4.7', 3.5, 4.7, '#7b3294'),
          ('burst        4.9-5.7', 4.9, 5.7, '#e66101'),
          ('quiet after  6.6-8.0', 6.6, 8.0, '#018571')]

b = np.load(C.OUT_DIR + '/birth_orbits_agama.npz')
ids, tf, eps = b['ids'], b['tform'], b['eps_birth']

# Initial masses, cached: the catalogue stores present-day mass, which has lost
# an age-dependent fraction to winds and would tilt the old end of the history.
ipath = C.OUT_DIR + '/insitu_imass.npz'
if os.path.exists(ipath):
    q = np.load(ipath); assert np.array_equal(q['ids'], ids); mi = q['imass']
else:
    from auriga_public import snapshot as snap_mod
    s = snap_mod.load_snapshot(127, 4, snappath=C.SIM_DIR,
        loadlist=['ParticleIDs', 'GFM_InitialMass', 'GFM_StellarFormationTime'])
    real = s.data['GFM_StellarFormationTime'] > 0
    sid = s.data['ParticleIDs'][real]
    im = s.data['GFM_InitialMass'][real] * C.MASS_TO_MSUN
    o = np.argsort(sid); ss = sid[o]
    p = np.searchsorted(ss, ids)
    ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == ids)
    mi = np.full(len(ids), np.nan, np.float32); mi[ok] = im[o[p[ok]]]
    np.savez(ipath, ids=ids, imass=mi)

good = np.isfinite(eps) & np.isfinite(mi)
tf, eps, mi = tf[good], eps[good], mi[good]
disc, halo = eps >= EPS_CUT, eps < EPS_CUT

bins = np.arange(np.floor(tf.min() * 10) / 10, C.T0_GYR + DT, DT)
ctr = .5 * (bins[:-1] + bins[1:])
sfr_a = np.histogram(tf, bins=bins, weights=mi)[0] / (DT * 1e9)
sfr_d = np.histogram(tf[disc], bins=bins, weights=mi[disc])[0] / (DT * 1e9)
sfr_h = np.histogram(tf[halo], bins=bins, weights=mi[halo])[0] / (DT * 1e9)
fh = np.divide(sfr_h, sfr_a, out=np.full_like(sfr_h, np.nan), where=sfr_a > 0)

print(f'measured near birth: {good.sum():,} stars, t = {tf.min():.2f}-{tf.max():.2f} Gyr')
print(f'\n{"epoch":24s} {"N":>9s} {"M_init":>10s} {"halo-born":>10s} {"disc-born":>10s} '
      f'{"med eps":>8s} {"SFR":>7s}')
rows = [(lab, (tf >= lo) & (tf < hi)) for lab, lo, hi, _ in EPOCHS]
rows.append(('Eos window %.2f-%.2f' % T_WIN, (tf >= T_WIN[0]) & (tf <= T_WIN[1])))
rows.append(('all measured', np.ones(len(tf), bool)))

for lab, m in rows:
    mh = mi[m & halo].sum() / mi[m].sum()
    dt = (tf[m].max() - tf[m].min()) if m.sum() else np.nan
    print(f'{lab:24s} {m.sum():9,} {mi[m].sum():10.3e} '
          f'{100 * mh:9.1f}% {100 * (1 - mh):9.1f}% {np.median(eps[m]):8.2f} '
          f'{mi[m].sum() / (dt * 1e9):7.1f}')

pk = np.nanargmax(sfr_a)
print(f'\nat the SFR peak (t = {ctr[pk]:.2f} Gyr): total {sfr_a[pk]:.1f}, '
      f'disc-born {sfr_d[pk]:.1f}, halo-born {sfr_h[pk]:.1f} Msun/yr '
      f'({100 * fh[pk]:.1f} per cent halo-born)')
q = (ctr > 3.0) & (ctr < 4.5)
print(f'pre-merger baseline 3.0-4.5 Gyr: total {sfr_a[q].mean():.1f}, '
      f'disc-born {sfr_d[q].mean():.1f}, halo-born {sfr_h[q].mean():.1f} Msun/yr '
      f'({100 * np.nanmean(fh[q]):.1f} per cent halo-born)')
print(f'halo-born SFR rises by x{sfr_h[pk] / sfr_h[q].mean():.2f}, '
      f'disc-born by x{sfr_d[pk] / sfr_d[q].mean():.2f}')

# ------------------------------------------------------------------ figure --
fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.0))


def mark(ax, label=False):
    ax.axvspan(*T_WIN, color=cM, alpha=.10, lw=0)
    ax.axvspan(T_COAL_LO, T_COAL_HI, color=cM, alpha=.45, lw=0,
               label='GS/E coalescence' if label else None)
    ax.axvline(T_PERI, color=cM, ls='--', lw=1.5,
               label='pericentre plunge' if label else None)


# (a) the history, split by birth orbit
ax = axes[0, 0]
mark(ax, label=True)
ax.fill_between(ctr, 0, sfr_h, step='mid', color=cH, alpha=.55, lw=0,
                label='born on halo orbits ($\\epsilon<0.5$)')
ax.fill_between(ctr, sfr_h, sfr_a, step='mid', color=cD, alpha=.45, lw=0,
                label='born on disc orbits ($\\epsilon\\geq0.5$)')
ax.step(ctr, sfr_a, where='mid', color='k', lw=1.4, label='total')
ax.step(ctr, sfr_h, where='mid', color=cH, lw=1.4)
ax.set(xlim=(tf.min(), C.T0_GYR), ylim=(0, 1.1 * np.nanmax(sfr_a)),
       xlabel='cosmic time [Gyr]', ylabel='SFR [M$_\\odot$ yr$^{-1}$]',
       title='(a) Star formation split by the orbit the star was born on')
ax.legend(fontsize=8.5, loc='upper right')

# (b) does the burst change the mix, or just the rate?
ax = axes[0, 1]
mark(ax)
ax.plot(ctr, 100 * fh, color=cH, lw=2.2)
ax.set(xlim=(tf.min(), C.T0_GYR), ylim=(0, None), xlabel='cosmic time [Gyr]',
       ylabel='per cent of newborn mass on halo orbits',
       title='(b) Halo-born fraction of star formation')
axr = ax.twinx()
axr.step(ctr, sfr_h, where='mid', color='.45', lw=1.2, ls='--')
axr.set_ylabel('halo-born SFR [M$_\\odot$/yr]', color='.45', fontsize=9)
axr.tick_params(axis='y', labelcolor='.45', labelsize=8)
axr.set_ylim(0, None)

# (c) the full distribution, no cut imposed
ax = axes[1, 0]
h = ax.hist2d(tf, eps, bins=[np.arange(tf.min(), C.T0_GYR, .1), np.linspace(-1.2, 1.4, 130)],
              cmap='viridis', cmin=1, norm='log')
ax.axhline(EPS_CUT, color='w', lw=1.2, ls='--')
ax.axhline(0, color='w', lw=.6, alpha=.5)
for t_, ls in [(T_PERI, '--'), (T_COAL_LO, '-'), (T_COAL_HI, '-')]:
    ax.axvline(t_, color=cM, ls=ls, lw=1.4)
ax.set(xlabel='cosmic time [Gyr]', ylabel='$\\epsilon$ at birth',
       title='(c) Birth circularity against birth time')
fig.colorbar(h[3], ax=ax, pad=.01, label='stars per bin')

# (d) the same information as distributions, epoch by epoch
ax = axes[1, 1]
eb = np.linspace(-1.2, 1.4, 90)
for lab, lo, hi, c in EPOCHS:
    m = (tf >= lo) & (tf < hi)
    ax.hist(eps[m], bins=eb, weights=mi[m], density=True, histtype='step', lw=2,
            color=c, label=f'{lab}  ({100 * mi[m & halo].sum() / mi[m].sum():.0f}% halo)')
ax.axvline(EPS_CUT, color='.3', ls='--', lw=1.2)
ax.set(xlabel='$\\epsilon$ at birth', ylabel='normalised density (mass-weighted)',
       title='(d) Birth circularity by epoch')
ax.legend(fontsize=8.5, loc='upper left')

fig.suptitle('Au18: the orbits stars are born on, through the GS/E merger', y=.985)
fig.tight_layout(rect=[0, 0, 1, .945])
out = C.FIG_DIR + '/au18_birth_orbit_sfh.png'
fig.savefig(out, dpi=150)
np.savez(C.OUT_DIR + '/birth_orbit_sfh.npz', t_bin=ctr, sfr_all=sfr_a,
         sfr_disc=sfr_d, sfr_halo=sfr_h, frac_halo=fh, eps_cut=EPS_CUT)
print('\nsaved', out)
