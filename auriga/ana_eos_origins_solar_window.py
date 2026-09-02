"""What a solar-neighbourhood observer would see of the two Eos populations.

The all-sky comparison (ana_eos_origins_actions.py) uses every star in each
population, which no observer has.  Here the Sun is placed at R = 8.1 kpc in the
disc plane and only stars within d < 4 kpc of it are kept.

That selection is severe.  Both populations are centrally concentrated -- the
disc-born one has median R = 3.9 kpc -- so a 4 kpc sphere at 8.1 kpc catches only
the outer tail of each: roughly 90 halo-born and 70 disc-born stars out of 4,283
and 3,300.  The azimuth of the Sun matters too, because the disc is not
axisymmetric; eight azimuths are run and the spread between them is reported.

Top row is the full population, bottom row the solar-neighbourhood selection, so
the effect of the selection can be read directly.

Both rows are drawn as kernel density estimates rather than histograms, because
the local samples are only ~70-100 stars and the binning dominated the shape.
Two things are done to keep that honest.  The kernels are reflected at the hard
edges of the selection -- |v_phi| < 80, ecc > 0.6, and the physical floors
J_R > 0 and J_R/|L_z| > 0 -- since a plain KDE spills across them and invents
density where the sample cannot reach.  And the bottom row carries a bootstrap
band, the 16th-84th percentile of 400 resamplings, which is the honest width of
what ~100 stars constrain; the top row shows a faint histogram behind the curve
so the KDE can be checked where the sample is large.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, gaussian_kde
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_au18 as C
import eos_origins_window as EO

os.makedirs(C.FIG_DIR, exist_ok=True)
RSUN, DMAX, NAZ = 8.1, 4.0, 8
d = EO.load()
cat = d['cat']
act = np.load(C.OUT_DIR + '/z0_actions.npz')
xy = np.load(C.OUT_DIR + '/z0_xy.npz')

o = np.argsort(act['ids']); aid = act['ids'][o]
p = np.searchsorted(aid, d['ids']); ok = (p < len(aid)) & (aid[np.minimum(p, len(aid) - 1)] == d['ids'])
ix = o[p[ok]]
for key in ('Jr', 'Jphi'):
    col = np.full(len(d['ids']), np.nan); col[ok] = act[key][ix]; d[key] = col
d['JrLz'] = d['Jr'] / np.abs(d['Jphi'])

ox = np.argsort(xy['ids']); xid = xy['ids'][ox]
px = np.searchsorted(xid, d['ids']); okx = (px < len(xid)) & (xid[np.minimum(px, len(xid) - 1)] == d['ids'])
ixx = ox[px[okx]]
X = np.full(len(d['ids']), np.nan); Y = np.full(len(d['ids']), np.nan); Z = np.full(len(d['ids']), np.nan)
X[okx] = xy['x'][ixx]; Y[okx] = xy['y'][ixx]; Z[okx] = xy['z'][ixx]


def solar_mask(phi):
    sx, sy = RSUN * np.cos(phi), RSUN * np.sin(phi)
    return np.sqrt((X - sx) ** 2 + (Y - sy) ** 2 + Z ** 2) < DMAX


AZ = np.linspace(0, 2 * np.pi, NAZ, endpoint=False)
sun0 = solar_mask(AZ[0])

C_HALO, C_DISC = '#7b3294', 'crimson'
fin = np.isfinite(d['Jr']) & np.isfinite(d['ecc'])
# bounds are the hard edges of the selection or of the quantity itself
PANELS = [('zvphi', r'$v_\phi$ [km s$^{-1}$]', np.linspace(-100, 100, 31), 'Azimuthal velocity', (-80., 80.)),
          ('ecc', 'eccentricity', np.linspace(0.55, 1.0, 25), 'Eccentricity', (0.6, 1.0)),
          ('Jr', r'$J_R$ [kpc km s$^{-1}$]', np.linspace(0, 3000, 27), 'Radial action', (0., None)),
          ('JrLz', r'$J_R/|L_z|$', np.linspace(0, 12, 25), '$J_R/|L_z|$', (0., None))]


def kde_reflected(v, grid, bounds, bw=None):
    """Gaussian KDE with reflection at hard bounds, so no density leaks past them.

    The bandwidth uses a ROBUST scale, min(sigma, IQR/1.349), rather than scipy's
    default.  J_R/|L_z| has a heavy tail -- the standard deviation is 160 and 613
    for the two populations against an interquartile range of order unity -- and
    Scott's rule driven by that sigma smears the distribution into a flat line
    that bears no resemblance to the histogram.
    """
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 5 or np.std(v) == 0:
        return None
    if bw is None:
        sig = np.std(v)
        iqr = np.subtract(*np.percentile(v, [75, 25]))
        scale = min(sig, iqr / 1.349) if iqr > 0 else sig
        bw = 0.9 * scale * len(v) ** (-0.2) / sig      # scipy multiplies by sigma
    k = gaussian_kde(v, bw_method=bw)
    out = k(grid)
    lo, hi = bounds
    if lo is not None:
        out += k(2 * lo - grid)
    if hi is not None:
        out += k(2 * hi - grid)
    if lo is not None:
        out[grid < lo] = 0.
    if hi is not None:
        out[grid > hi] = 0.
    return out


def kde_band(v, grid, bounds, nboot=400, rng=None):
    """16th-84th percentile of the KDE over bootstrap resamplings."""
    rng = rng or np.random.default_rng(11)
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    curves = []
    for _ in range(nboot):
        c = kde_reflected(rng.choice(v, len(v), replace=True), grid, bounds)
        if c is not None:
            curves.append(c)
    if not curves:
        return None, None
    return np.percentile(curves, [16, 84], axis=0)

fig, axes = plt.subplots(2, 4, figsize=(23, 10.4))
for row, (sel, tag) in enumerate([(np.ones(len(d['ids']), bool), 'all stars in each population'),
                                  (sun0, f'$d<{DMAX:.0f}$ kpc of a Sun at $R={RSUN}$ kpc')]):
    for col, (key, xlab, bins, title, bounds) in enumerate(PANELS):
        ax = axes[row, col]
        grid = np.linspace(bins[0], bins[-1], 400)
        for lab, m, c in [('halo-born (merger-triggered)', d['halo_born'], C_HALO),
                          ('disc-born (heated)', d['disc_born'], C_DISC)]:
            v = d[key][m & fin & sel]
            v = v[np.isfinite(v)]
            if len(v) < 5:
                continue
            if row == 0:                       # faint histogram to check the KDE against
                ax.hist(v, bins=bins, density=True, histtype='step', lw=1.0,
                        color=c, alpha=.35)
            curve = kde_reflected(v, grid, bounds)
            if curve is not None:
                ax.plot(grid, curve, color=c, lw=2.4, label=f'{lab} ({len(v):,})')
            if row == 1:
                lo, hi = kde_band(v, grid, bounds)
                if lo is not None:
                    ax.fill_between(grid, lo, hi, color=c, alpha=.18, lw=0)
                ax.plot(v, np.full(len(v), -0.03 * ax.get_ylim()[1]), '|', color=c,
                        ms=6, alpha=.5)
            ax.axvline(np.median(v), color=c, ls=':', lw=1.5)
        ax.set(xlabel=xlab, ylabel='density', xlim=(bins[0], bins[-1]),
               title=f'({"abcd efgh"[row * 5 + col]}) {title}')
        ax.legend(fontsize=8.5)
    axes[row, 0].text(.02, .98, tag, transform=axes[row, 0].transAxes, va='top',
                      fontsize=10, style='italic',
                      bbox=dict(fc='white', alpha=.85, ec='none'))

fig.suptitle(f'Au18: the two Eos populations, all stars (top) and as seen from a Sun at '
             f'$R={RSUN}$ kpc within $d<{DMAX:.0f}$ kpc (bottom).  Kernel density estimates, '
             f'reflected at the selection bounds;\nshaded band = 16th-84th percentile over 400 '
             f'bootstrap resamplings, ticks = the individual stars', fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, .95])
out = C.FIG_DIR + '/au18_eos_origins_solar_win.png'
fig.savefig(out, dpi=145)

print(f'Sun at R={RSUN} kpc, d<{DMAX} kpc, {NAZ} azimuths\n')
print(f'{"":12s} {"all: halo":>10s} {"disc":>7s} | {"solar: halo":>12s} {"disc":>7s}')
for lab, m in [('N', None)]:
    pass
nh = [(d['halo_born'] & fin & solar_mask(a)).sum() for a in AZ]
nd = [(d['disc_born'] & fin & solar_mask(a)).sum() for a in AZ]
print(f'{"N":12s} {(d["halo_born"] & fin).sum():10,} {(d["disc_born"] & fin).sum():7,} | '
      f'{int(np.median(nh)):12,} {int(np.median(nd)):7,}   '
      f'(range over azimuth {min(nh)}-{max(nh)} and {min(nd)}-{max(nd)})')
print()
for key, xlab, bins, title, bounds in PANELS:
    a_all = d[key][d['halo_born'] & fin]; b_all = d[key][d['disc_born'] & fin]
    a_all = a_all[np.isfinite(a_all)]; b_all = b_all[np.isfinite(b_all)]
    ks_all = ks_2samp(a_all, b_all)
    ds, meds_h, meds_d = [], [], []
    for a_phi in AZ:
        sm = solar_mask(a_phi)
        aa = d[key][d['halo_born'] & fin & sm]; bb = d[key][d['disc_born'] & fin & sm]
        aa = aa[np.isfinite(aa)]; bb = bb[np.isfinite(bb)]
        if len(aa) > 10 and len(bb) > 10:
            ds.append(ks_2samp(aa, bb).statistic)
            meds_h.append(np.median(aa)); meds_d.append(np.median(bb))
    print(f'{key:6s}  all-sky: halo {np.median(a_all):8.2f}  disc {np.median(b_all):8.2f}  '
          f'KS D={ks_all.statistic:.3f}  |  solar: halo {np.median(meds_h):8.2f}  '
          f'disc {np.median(meds_d):8.2f}  KS D={np.median(ds):.3f} '
          f'(azimuth spread {np.min(ds):.2f}-{np.max(ds):.2f})')
print('\nsaved', out)
