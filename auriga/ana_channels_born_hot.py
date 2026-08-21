"""Au18: born hot or heated?  The A/B/C channels in birth and present-day kinematics.

ana_eos_age_kinematics.py selects the Eos analogue the way the observations do,
from present-day orbits.  This is the test the simulation can do and the data
cannot: follow the same stars back to birth and ask whether their hot orbits are
what they were born with.

  A  heated disc      born cold in the plane during the merger, hot today
  B  born radial      born hot off-plane during the merger
  C  pre-merger Splash born cold in the plane *before* the merger, hot today

A and C are the heating channels, B is the born-hot (onset) channel.  The last
two panels close the loop with the observational selection: they show where each
channel lands at z=0 and what fraction of it a Milky Way observer would actually
count as Eos.

Reads out/eos_two_channels.npz, out/merger_birth_radii.npz, out/premerger_splash.npz
and out/z0_insitu_catalog.npz.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import orbit_tools as OT
import config_au18 as C
import channels_au18 as ch_mod

os.makedirs(C.FIG_DIR, exist_ok=True)
VPHI_MAX, ECC_MIN = 100., 0.6         # the observational Eos box, as in ana_eos_age_kinematics
T_COAL = 5.4
cA, cB, cC = '#2166ac', '#762a83', '#e08214'

d = ch_mod.load()
A, B, base = d['A'], d['B'], d['base']

sp = np.load(C.OUT_DIR + '/premerger_splash.npz')
sp_ok = np.isfinite(sp['eps_birth']) & np.isfinite(sp['eps_z0']) & np.isfinite(sp['z_birth'])
Csel = sp_ok & (sp['eps_birth'] > .7) & (sp['z_birth'] < ch_mod.Z_A_MAX) & (sp['eps_z0'] < .3)

CH = [('A: heated disc', d['ids'][A], d['eps_birth'][A], d['eps_z0'][A], d['tform'][A], cA),
      ('B: born radial', d['ids'][B], d['eps_birth'][B], d['eps_z0'][B], d['tform'][B], cB),
      ('C: pre-merger Splash', sp['ids'][Csel], sp['eps_birth'][Csel], sp['eps_z0'][Csel],
       sp['tform'][Csel], cC)]

# --- present-day orbits of each channel, from the catalogue -------------------
cat = np.load(C.OUT_DIR + '/z0_insitu_catalog.npz')
order = np.argsort(cat['ids'])
sorted_ids = cat['ids'][order]


def z0_of(ids):
    p = np.searchsorted(sorted_ids, ids)
    ok = (p < len(sorted_ids)) & (sorted_ids[np.minimum(p, len(sorted_ids) - 1)] == ids)
    ix = order[p[ok]]
    return {k: cat[k][ix] for k in ('vphi', 'ecc', 'rapo', 'r', 'R', 'age', 'feh')}, ok


fig, axes = plt.subplots(2, 3, figsize=(18.5, 10.4))

# (a) the birth-vs-now plane for the merger-born sample ------------------------
ax = axes[0, 0]
ax.hist2d(d['eps_birth'][base], d['eps_z0'][base], bins=(90, 90), range=((-1, 1.4), (-1, 1.4)),
          norm=LogNorm(), cmap='Greys')
ax.plot([-1, 1.4], [-1, 1.4], color='k', lw=.8, ls=':')
ax.add_patch(Rectangle((.7, -1), .7, 1.3, fill=False, ec=cA, lw=2))
ax.add_patch(Rectangle((-1, -1), 1.3, 1.3, fill=False, ec=cB, lw=2))
ax.text(1.0, -.85, 'A', color=cA, fontsize=13, weight='bold', ha='center')
ax.text(-.85, -.85, 'B', color=cB, fontsize=13, weight='bold', ha='center')
ax.set(xlabel=r'$\epsilon$ at birth', ylabel=r'$\epsilon$ at $z=0$',
       title='(a) Merger-born in-situ stars: birth vs now\n(dotted = unchanged circularity)')
ax.text(.02, .02, 'boxes show the $\\epsilon$ criteria only;\nA and B also carry $|z_{\\rm birth}|$ cuts',
        transform=ax.transAxes, fontsize=7.5, color='.25')

# (b) how much each channel was heated ----------------------------------------
ax = axes[0, 1]
bins = np.linspace(-1.6, 1.0, 45)
for lab, ids, eb, e0, tf, c in CH:
    ax.hist(e0 - eb, bins=bins, density=True, histtype='step', lw=2, color=c,
            label=f'{lab} (N={len(ids):,})')
    ax.axvline(np.median(e0 - eb), color=c, ls=':', lw=1.4)
ax.axvline(0, color='k', lw=.8)
ax.set(xlabel=r'$\Delta\epsilon = \epsilon_{z=0} - \epsilon_{\rm birth}$',
       ylabel='normalised density', title='(b) Change in circularity since birth')
ax.legend(fontsize=8.5, loc='upper left')

# (c) when each channel formed -------------------------------------------------
ax = axes[0, 2]
tmin = min(tf.min() for *_, tf, _ in CH)
tmax = max(tf.max() for *_, tf, _ in CH)
bins = np.linspace(tmin, tmax, 45)
for lab, ids, eb, e0, tf, c in CH:
    ax.hist(tf, bins=bins, density=True, histtype='step', lw=2, color=c, label=lab)
    ax.axvline(np.median(tf), color=c, ls=':', lw=1.4)
ax.axvline(T_COAL, color='goldenrod', lw=2, label='coalescence')
sec = ax.secondary_xaxis('top', functions=(lambda t: C.T0_GYR - t, lambda a: C.T0_GYR - a))
sec.set_xlabel('age [Gyr]')
ax.set(xlabel='birth cosmic time [Gyr]', ylabel='normalised density',
       title='(c) When each channel formed')
ax.text(.02, .70, 'the windows differ by construction:\nA, B are merger-born, C pre-merger',
        transform=ax.transAxes, fontsize=7.5, color='.25')
ax.legend(fontsize=8.5)

# (d) present-day orbits -------------------------------------------------------
ax = axes[1, 0]
ins = (cat['R'] > 4) & (cat['R'] < 30) & np.isfinite(cat['ecc'])
ax.hist2d(cat['vphi'][ins], cat['ecc'][ins], bins=(110, 90), range=((-250, 400), (0, 1)),
          norm=LogNorm(), cmap='Greys')
ax.add_patch(Rectangle((-VPHI_MAX, ECC_MIN), 2 * VPHI_MAX, 1 - ECC_MIN,
                       fill=False, ec='k', lw=2, ls='--'))
frac = {}
for lab, ids, eb, e0, tf, c in CH:
    z0, ok = z0_of(ids)
    OT.density_contours(ax, z0['vphi'], z0['ecc'], [[-250, 400], [0, 1]], c,
                        label=lab, levels=(0.8, 0.4), bins=60)
    inbox = (np.abs(z0['vphi']) < VPHI_MAX) & (z0['ecc'] > ECC_MIN)
    frac[lab] = (inbox.mean(), len(z0['vphi']), np.median(z0['rapo']), np.median(z0['feh']))
ax.text(0, .97, 'observational\nEos box', ha='center', va='top', fontsize=8)
ax.set(xlabel=r'$v_\phi$ [km s$^{-1}$]', ylabel='eccentricity',
       title='(d) Where each channel sits at $z=0$')
ax.legend(fontsize=8.5, loc='lower left')

# (e) what an observer would actually select -----------------------------------
ax = axes[1, 1]
labs = [l for l, *_ in CH]
vals = [100 * frac[l][0] for l in labs]
ax.bar(range(len(labs)), vals, color=[c for *_, c in CH], alpha=.85)
for i, (l, v) in enumerate(zip(labs, vals)):
    ax.text(i, v + 1, f'{v:.0f}%\n(N={frac[l][1]:,})', ha='center', fontsize=9)
ax.set_xticks(range(len(labs)))
ax.set_xticklabels([l.split(':')[0] + '\n' + l.split(': ')[1] for l in labs], fontsize=9)
ax.set(ylabel='per cent selected as Eos by the observational cut',
       title=f'(e) Recovered by $|v_\\phi|<{VPHI_MAX:.0f}$, ecc$>{ECC_MIN:.1f}$',
       ylim=(0, max(vals) * 1.25))

# (f) apocentre of each channel ------------------------------------------------
ax = axes[1, 2]
b = np.linspace(0, 40, 41)
for lab, ids, eb, e0, tf, c in CH:
    z0, ok = z0_of(ids)
    ax.hist(z0['rapo'], bins=b, density=True, histtype='step', lw=2, color=c, label=lab)
g_ok = (cat['gse_R'] > 4) & (cat['gse_R'] < 30) & np.isfinite(cat['gse_ecc'])
ax.hist(cat['gse_rapo'][g_ok], bins=b, density=True, histtype='step', lw=1.8, ls='--',
        color='#1b7837', label='GS/E debris')
ax.set(xlabel=r'$r_{\rm apo}$ [kpc]', ylabel='normalised', title='(f) Apocentre at $z=0$')
ax.legend(fontsize=8.5)

fig.suptitle('Au18: born hot or heated?  The three in-situ channels in birth and present-day kinematics',
             fontsize=14)
fig.tight_layout(rect=[0, 0, 1, .95])
out = C.FIG_DIR + '/au18_channels_born_hot.png'
fig.savefig(out, dpi=140)

print(f'{"channel":24s} {"N":>7s} {"med d_eps":>10s} {"med t_birth":>12s} '
      f'{"% in Eos box":>13s} {"med r_apo":>10s} {"med [Fe/H]":>11s}')
for lab, ids, eb, e0, tf, c in CH:
    f_, n_, ra_, fe_ = frac[lab]
    print(f'{lab:24s} {len(ids):7,} {np.median(e0 - eb):10.2f} {np.median(tf):12.2f} '
          f'{100*f_:12.1f}% {ra_:10.1f} {fe_:+11.2f}')
print('\nsaved', out)
