"""Can the two Eos populations be told apart from present-day kinematics alone?

The two are defined by their BIRTH azimuthal velocity, which no observer has.
This asks whether anything measurable today separates them: v_phi, eccentricity,
the radial action J_R, and the ratio J_R/|L_z|.

Actions come from AGAMA's ActionFinder run on an axisymmetric potential fitted to
the z=0 snapshot (prep_actions.py).  J_R is the cleaner heat measure of the four,
being an adiabatic invariant, whereas eccentricity depends on the potential's
shape and v_phi alone conflates heat with angular momentum.  J_R/|L_z| is
dimensionless and so comparable between galaxies.

Reads out/z0_actions.npz and the cached catalogues.
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
d = EO.load()
cat = d['cat']
act = np.load(C.OUT_DIR + '/z0_actions.npz')

# match the actions onto the merger-born sample
o = np.argsort(act['ids']); aid = act['ids'][o]
p = np.searchsorted(aid, d['ids'])
ok = (p < len(aid)) & (aid[np.minimum(p, len(aid) - 1)] == d['ids'])
ix = o[p[ok]]
for key in ('Jr', 'Jz', 'Jphi'):
    col = np.full(len(d['ids']), np.nan)
    col[ok] = act[key][ix]
    d[key] = col
d['JrLz'] = d['Jr'] / np.abs(d['Jphi'])

g = np.isfinite(act['gse_Jr'])
gse = dict(vphi=cat['gse_vphi'][g], ecc=cat['gse_ecc'][g], Jr=act['gse_Jr'][g],
           JrLz=act['gse_Jr'][g] / np.abs(act['gse_Jphi'][g]))

C_HALO, C_DISC, C_GSE = '#7b3294', 'crimson', '#1f6fd0'
fin = np.isfinite(d['Jr']) & np.isfinite(d['ecc'])
POPS = [('halo-born Eos (merger-triggered)', d['halo_born'] & fin, C_HALO),
        ('disc-born Eos (heated)', d['disc_born'] & fin, C_DISC)]

PANELS = [
    ('vphi', r'$v_\phi$ [km s$^{-1}$]', np.linspace(-150, 150, 46), False, '(a) Azimuthal velocity'),
    ('ecc', 'eccentricity', np.linspace(0.4, 1.0, 40), False, '(b) Eccentricity'),
    ('Jr', r'$J_R$ [kpc km s$^{-1}$]', np.linspace(0, 4000, 46), False, '(c) Radial action'),
    ('JrLz', r'$J_R / |L_z|$', np.linspace(0, 15, 46), False, '(d) $J_R/|L_z|$'),
]
KEY = {'vphi': 'zvphi', 'ecc': 'ecc', 'Jr': 'Jr', 'JrLz': 'JrLz'}

fig, axes = plt.subplots(1, 4, figsize=(23, 5.6))
stats = {}
for ax, (key, xlab, bins, logx, title) in zip(axes, PANELS):
    for lab, m, c in POPS:
        v = d[KEY[key]][m]
        v = v[np.isfinite(v)]
        ax.hist(v, bins=bins, density=True, histtype='step', lw=2.3, color=c,
                label=f'{lab} ({m.sum():,})')
        ax.axvline(np.median(v), color=c, ls=':', lw=1.5)
        stats.setdefault(key, {})[lab] = v
    gv = gse[key]
    gv = gv[np.isfinite(gv)]
    ax.hist(gv, bins=bins, density=True, histtype='step', lw=1.7, ls='--', color=C_GSE,
            label='GS/E debris')
    ax.set(xlabel=xlab, ylabel='normalised density', title=title,
           xlim=(bins[0], bins[-1]))
    ax.legend(fontsize=8)

fig.suptitle('Au18: present-day kinematics of the two Eos populations -- is there any '
             'observable that separates them?', fontsize=13.5)
fig.tight_layout(rect=[0, 0, 1, .93])
out = C.FIG_DIR + '/au18_eos_origins_actions.png'
fig.savefig(out, dpi=145)

print(f'{"":14s} {"halo-born":>22s} {"disc-born":>22s} {"KS D":>7s} {"p":>10s} {"overlap":>9s}')
for key, xlab, bins, logx, title in PANELS:
    a = stats[key]['halo-born Eos (merger-triggered)']
    b = stats[key]['disc-born Eos (heated)']
    ks = ks_2samp(a, b)
    # overlap coefficient of the two normalised histograms
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    w = np.diff(bins)
    ov = np.sum(np.minimum(ha, hb) * w)
    print(f'{key:14s} {np.median(a):10.3f} +/- {np.std(a):7.3f} '
          f'{np.median(b):10.3f} +/- {np.std(b):7.3f} '
          f'{ks.statistic:7.3f} {ks.pvalue:10.2e} {ov:9.3f}')
print('\n(overlap = fraction of the two distributions that coincides; 1 = identical)')
print('\nfraction beyond the right-hand edge of each linear axis:')
for key, xlab, bins, logx, title in PANELS:
    frac = {lab: np.mean(v > bins[-1]) for lab, v in stats[key].items()}
    gv = gse[key][np.isfinite(gse[key])]
    print(f'  {key:6s} > {bins[-1]:7.1f}:  '
          + '  '.join(f'{lab.split()[0]} {f:.3f}' for lab, f in frac.items())
          + f'  GS/E {np.mean(gv > bins[-1]):.3f}')
print('saved', out)
