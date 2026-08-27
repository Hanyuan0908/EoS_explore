"""The full age-metallicity plane of Au18, with the two Eos populations on it.

Everything in-situ at z=0 (1.98M stars) forms the background, so the two Eos
populations can be seen against the galaxy's whole enrichment history rather than
just against the merger-born stars they were drawn from.

The GS/E merger is marked twice over: the shaded band spans our own dating of the
event, from the pericentre plunge at t = 5.0 Gyr to coalescence at 5.4
(auriga/PROGRESS.md), and the solid line is the 5.23 Gyr that Fattahi et al.
(2019) report for this halo.  On an age axis those become ages of 8.4-8.8 and
8.59 Gyr.

Contours: disc-born Eos (born on the disc ridge, heated later) and halo-born Eos
(born already hot in the plunge, i.e. the merger-triggered population), split at
v_phi,birth = 150 km/s by eos_origins.py.  GS/E debris is drawn for reference
since it occupies its own corner of this plane.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import orbit_tools as OT
import config_au18 as C
import eos_origins as EO

os.makedirs(C.FIG_DIR, exist_ok=True)
d = EO.load()
cat = d['cat']

T_PLUNGE, T_COAL, T_FATTAHI = 5.0, 5.4, 5.23
AGE_PLUNGE, AGE_COAL = C.T0_GYR - T_PLUNGE, C.T0_GYR - T_COAL
AGE_FATTAHI = C.T0_GYR - T_FATTAHI
C_HALO, C_DISC, C_GSE = '#7b3294', 'crimson', '#1f6fd0'

age_all, feh_all = cat['age'], cat['feh']
good = np.isfinite(age_all) & np.isfinite(feh_all)
g_ok = np.isfinite(cat['gse_age']) & np.isfinite(cat['gse_feh'])

VIEWS = [dict(xlim=(0, 13.8), ylim=(-2.5, 0.7), title='All in-situ stars'),
         dict(xlim=(7.0, 10.5), ylim=(-1.6, 0.5), title='Zoom on the merger epoch')]
fig, axes = plt.subplots(1, 2, figsize=(17, 6.4))

for ax, v in zip(axes, VIEWS):
    ax.hist2d(age_all[good], feh_all[good], bins=(200, 150),
              range=(v['xlim'], v['ylim']), norm=LogNorm(), cmap='Greys')
    ax.axvspan(AGE_COAL, AGE_PLUNGE, color='goldenrod', alpha=.22, lw=0)
    ax.axvline(AGE_FATTAHI, color='goldenrod', lw=2.2)
    rng = [list(v['xlim']), list(v['ylim'])]
    OT.density_contours(ax, cat['gse_age'][g_ok], cat['gse_feh'][g_ok], rng, C_GSE,
                        label=f'GS/E debris ({g_ok.sum():,})', levels=(0.9, 0.5), bins=70, ls='--')
    OT.density_contours(ax, d['age'][d['disc_born']], d['feh'][d['disc_born']], rng, C_DISC,
                        label=f"disc-born Eos ({d['disc_born'].sum():,})",
                        levels=(0.9, 0.6, 0.3), bins=60, lw=2.0)
    OT.density_contours(ax, d['age'][d['halo_born']], d['feh'][d['halo_born']], rng, C_HALO,
                        label=f"halo-born Eos, merger-triggered ({d['halo_born'].sum():,})",
                        levels=(0.9, 0.6, 0.3), bins=60, lw=2.0)
    ax.set(xlim=v['xlim'], ylim=v['ylim'], xlabel='age [Gyr]', ylabel='[Fe/H]',
           title=v['title'])
    sec = ax.secondary_xaxis('top', functions=(lambda a: C.T0_GYR - a, lambda t: C.T0_GYR - t))
    sec.set_xlabel('formation cosmic time [Gyr]')
    ax.text(AGE_FATTAHI, v['ylim'][0] + .06 * (v['ylim'][1] - v['ylim'][0]),
            '  GS/E merger', color='darkgoldenrod', fontsize=10, rotation=90, va='bottom')
    ax.legend(fontsize=9, loc='lower left')

fig.suptitle('Au18 age-metallicity plane: all in-situ stars (greyscale), the two Eos populations, '
             'and the GS/E merger\n'
             f'(band = plunge {T_PLUNGE} to coalescence {T_COAL} Gyr; line = {T_FATTAHI} Gyr, '
             'Fattahi et al. 2019)', fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, .90])
out = C.FIG_DIR + '/au18_eos_origins_amr.png'
fig.savefig(out, dpi=145)

print(f'merger: t = {T_PLUNGE}-{T_COAL} Gyr  ->  age {AGE_COAL:.2f}-{AGE_PLUNGE:.2f} Gyr')
print(f'{"":34s} {"N":>9s} {"age":>7s} {"[Fe/H]":>8s}')
for lab, m in [('all in-situ', good),
               ('disc-born Eos', d['disc_born']),
               ('halo-born Eos (merger-triggered)', d['halo_born'])]:
    a = age_all[m] if lab == 'all in-situ' else d['age'][m]
    f = feh_all[m] if lab == 'all in-situ' else d['feh'][m]
    print(f'{lab:34s} {m.sum():9,} {np.median(a):7.2f} {np.median(f):+8.2f}')
print(f'{"GS/E debris":34s} {g_ok.sum():9,} {np.median(cat["gse_age"][g_ok]):7.2f} '
      f'{np.nanmedian(cat["gse_feh"][g_ok]):+8.2f}')
print()
older = age_all[good] > AGE_PLUNGE
print(f'in-situ stars older than the plunge: {older.sum():,} '
      f'({100*older.mean():.1f}% of all in-situ), median [Fe/H] {np.median(feh_all[good][older]):+.2f}')
print(f'in-situ stars younger:               {(~older).sum():,}, '
      f'median [Fe/H] {np.median(feh_all[good][~older]):+.2f}')
print('saved', out)
