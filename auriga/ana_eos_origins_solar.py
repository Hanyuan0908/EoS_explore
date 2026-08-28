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
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_au18 as C
import eos_origins as EO

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
PANELS = [('zvphi', r'$v_\phi$ [km s$^{-1}$]', np.linspace(-150, 150, 31), 'Azimuthal velocity'),
          ('ecc', 'eccentricity', np.linspace(0.4, 1.0, 25), 'Eccentricity'),
          ('Jr', r'$J_R$ [kpc km s$^{-1}$]', np.linspace(0, 4000, 27), 'Radial action'),
          ('JrLz', r'$J_R/|L_z|$', np.linspace(0, 15, 25), '$J_R/|L_z|$')]

fig, axes = plt.subplots(2, 4, figsize=(23, 10.4))
for row, (sel, tag) in enumerate([(np.ones(len(d['ids']), bool), 'all stars in each population'),
                                  (sun0, f'$d<{DMAX:.0f}$ kpc of a Sun at $R={RSUN}$ kpc')]):
    for col, (key, xlab, bins, title) in enumerate(PANELS):
        ax = axes[row, col]
        for lab, m, c in [('halo-born (merger-triggered)', d['halo_born'], C_HALO),
                          ('disc-born (heated)', d['disc_born'], C_DISC)]:
            v = d[key][m & fin & sel]
            v = v[np.isfinite(v)]
            if len(v) < 5:
                continue
            ax.hist(v, bins=bins, density=True, histtype='step', lw=2.3, color=c,
                    label=f'{lab} ({len(v):,})')
            ax.axvline(np.median(v), color=c, ls=':', lw=1.5)
        ax.set(xlabel=xlab, ylabel='normalised density', xlim=(bins[0], bins[-1]),
               title=f'({"abcd efgh"[row * 5 + col]}) {title}')
        ax.legend(fontsize=8.5)
    axes[row, 0].text(.02, .98, tag, transform=axes[row, 0].transAxes, va='top',
                      fontsize=10, style='italic',
                      bbox=dict(fc='white', alpha=.85, ec='none'))

fig.suptitle(f'Au18: the two Eos populations, all stars (top) and as seen from a Sun at '
             f'$R={RSUN}$ kpc within $d<{DMAX:.0f}$ kpc (bottom)', fontsize=13.5)
fig.tight_layout(rect=[0, 0, 1, .95])
out = C.FIG_DIR + '/au18_eos_origins_solar.png'
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
for key, xlab, bins, title in PANELS:
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
