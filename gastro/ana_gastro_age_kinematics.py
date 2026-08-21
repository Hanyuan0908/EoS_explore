"""gastro / joaorun003: age and kinematic signature of the Eos-like population.

The companion to auriga/ana_eos_age_kinematics.py, laid out panel-for-panel so
the two simulations can be read side by side.  The observational result is that
Eos -- metal-poor low-alpha stars on non-rotating, eccentric orbits -- is *older
at fixed metallicity* than the rotating population of the same chemistry, i.e. it
formed before the disc spun up rather than being disc stars kicked onto halo
orbits later.  This asks the same of the gastro models using the discriminants
these snapshots carry: age, total metallicity and present-day orbits.

Caveats (see gastro_config for why):
  * total Z only, so "metal-poor" is [M/H] and there is no alpha split -- the Eos
    analogue here is kinematic, not low-alpha;
  * a single snapshot, so no birth kinematics: the born-hot vs heated test lives
    on the Auriga side;
  * the accreted-particle list is not available, so the merger debris cannot be
    labelled directly.  The merger epoch used below is measured from the snapshot
    itself -- the fraction of stars that are retrograde today spikes seven-fold
    for stars born at t = 1.5-2.25 Gyr, and the eccentric fraction turns back up
    at the same time.

Reads out/<model>_stars.npz (built by prep_gastro.py).
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
import gastro_config as G

os.makedirs(G.FIG_DIR, exist_ok=True)
RMIN, RMAX = 4., 30.
T_MERGE = (1.5, 2.25)                 # measured from the snapshot; see the docstring
C_EOS, C_DISC, C_ALL = '#b2182b', '#2166ac', '.55'


def panels(axes, d, label):
    R, vphi, ecc = d['R'], d['vphi'], d['ecc']
    tform, age, mh, rapo, Lz, E = d['tform'], d['age'], d['mh'], d['rapo'], d['Lz'], d['E']
    t_now = float(d['t_now'])
    ins = (R > RMIN) & (R < RMAX) & np.isfinite(ecc)
    eos = ins & (np.abs(vphi) < G.EOS_VPHI_MAX) & (ecc > G.EOS_ECC_MIN)
    disc = ins & (vphi > G.DISC_VPHI_MIN) & (ecc < G.DISC_ECC_MAX)
    tb = np.linspace(0, t_now, 45)
    ctr = .5 * (tb[:-1] + tb[1:])

    # (a) spin-up
    ax = axes[0]
    ax.hist2d(tform[ins], vphi[ins], bins=(130, 120), range=((0, t_now), (-250, 400)),
              norm=LogNorm(), cmap='Greys')
    ib = np.clip(np.searchsorted(tb, tform[ins]) - 1, 0, len(tb) - 2)
    vv = vphi[ins]
    med = np.array([np.median(vv[ib == k]) if (ib == k).sum() > 50 else np.nan
                    for k in range(len(tb) - 1)])
    ax.axvspan(*T_MERGE, color='goldenrod', alpha=.2, lw=0, label='merger')
    ax.plot(ctr, med, color='gold', lw=2.5, label=r'median $v_\phi$')
    ax.axhline(0, color='k', lw=.6, ls=':')
    ax.axhline(G.EOS_VPHI_MAX, color=C_EOS, lw=.9, ls='--')
    ax.axhline(-G.EOS_VPHI_MAX, color=C_EOS, lw=.9, ls='--')
    ax.set(xlabel='birth time [Gyr]', ylabel=r'$v_\phi$ [km s$^{-1}$]',
           title=f'{label}\n(a) Disc spin-up')
    ax.legend(fontsize=8.5, loc='lower right')

    # (b) selection in the orbit plane
    ax = axes[1]
    ax.hist2d(vphi[ins], ecc[ins], bins=(120, 100), range=((-250, 400), (0, 1)),
              norm=LogNorm(), cmap='Greys')
    ax.add_patch(Rectangle((-G.EOS_VPHI_MAX, G.EOS_ECC_MIN), 2 * G.EOS_VPHI_MAX,
                           1 - G.EOS_ECC_MIN, fill=False, ec=C_EOS, lw=2, label='Eos-like'))
    ax.add_patch(Rectangle((G.DISC_VPHI_MIN, 0), 400 - G.DISC_VPHI_MIN, G.DISC_ECC_MAX,
                           fill=False, ec=C_DISC, lw=2, label='disc orbits'))
    ax.set(xlabel=r'$v_\phi$ [km s$^{-1}$]', ylabel='eccentricity',
           title='(b) Selection in the orbit plane')
    ax.legend(fontsize=8.5, loc='lower left')

    # (c) ages
    ax = axes[2]
    bins = np.linspace(0, t_now, 45)
    ax.hist(age[ins], bins=bins, density=True, histtype='stepfilled', color=C_ALL, alpha=.3,
            label=f'all, {RMIN:.0f}<R<{RMAX:.0f} kpc (N={ins.sum():,})')
    for m, c, l in [(eos, C_EOS, f'Eos-like (N={eos.sum():,})'),
                    (disc, C_DISC, f'disc orbits (N={disc.sum():,})')]:
        ax.hist(age[m], bins=bins, density=True, histtype='step', lw=2, color=c, label=l)
        ax.axvline(np.median(age[m]), color=c, ls=':', lw=1.4)
    ax.axvspan(t_now - T_MERGE[1], t_now - T_MERGE[0], color='goldenrod', alpha=.2, lw=0)
    ax.set(xlabel='age [Gyr]', ylabel='normalised density', title='(c) Age by orbit type')
    ax.legend(fontsize=8)

    # (d) age at fixed metallicity
    ax = axes[3]
    edges = np.linspace(-1.0, 0.25, 14)
    cen = .5 * (edges[:-1] + edges[1:])

    def tr(mask):
        med, lo, hi, n = [], [], [], []
        for i in range(len(cen)):
            a = age[mask & (mh >= edges[i]) & (mh < edges[i + 1])]
            if len(a) >= 30:
                med.append(np.median(a)); lo.append(np.percentile(a, 25)); hi.append(np.percentile(a, 75))
            else:
                med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            n.append(len(a))
        return np.array(med), np.array(lo), np.array(hi), np.array(n)

    hm, hlo, hhi, hn = tr(eos)
    dm, dlo, dhi, dn = tr(disc)
    ax.fill_between(cen, hlo, hhi, color=C_EOS, alpha=.15)
    ax.fill_between(cen, dlo, dhi, color=C_DISC, alpha=.15)
    ax.plot(cen, hm, 's-', color=C_EOS, lw=2, ms=5, label='Eos-like (hot, eccentric)')
    ax.plot(cen, dm, 'o-', color=C_DISC, lw=2, ms=5, label='disc orbits')
    ax.set(xlabel='[M/H]', ylabel='median age [Gyr]', title='(d) Age at fixed metallicity')
    ax.legend(fontsize=8.5)
    both = np.isfinite(hm) & np.isfinite(dm)
    off = np.mean((hm - dm)[both])
    ax.text(.03, .06, f'mean offset {off:+.2f} Gyr', transform=ax.transAxes, fontsize=8.5,
            bbox=dict(fc='white', alpha=.85, ec='none'))

    # (e) formation history of the Eos analogue
    ax = axes[4]
    n_eos, _ = np.histogram(tform[eos], bins=tb)
    ntot, _ = np.histogram(tform[ins], bins=tb)
    ax.fill_between(ctr, 0, n_eos, step='mid', color=C_EOS, alpha=.35, lw=0)
    ax.step(ctr, n_eos, where='mid', color=C_EOS, lw=2, label='Eos-like births per bin')
    ax.axvspan(*T_MERGE, color='goldenrod', alpha=.2, lw=0)
    ax.set(xlabel='birth time [Gyr]', ylabel='Eos-like stars per bin',
           title='(e) Formation history of the Eos analogue')
    _in, _ctrl, _enh = OT.local_enhancement(tform, eos, ins, T_MERGE)
    ax.text(.97, .95, f'merger {T_MERGE[0]}-{T_MERGE[1]} Gyr:\n{100*_in:.0f}% of the cohort ends '
                      f'up\nEos-like vs {100*_ctrl:.0f}% either side (x{_enh:.1f})',
            transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
            bbox=dict(fc='white', alpha=.85, ec='none'))
    ax2 = ax.twinx()
    ok = ntot > 200
    ax2.plot(ctr[ok], 100 * n_eos[ok] / ntot[ok], color='k', lw=1.6, ls='--',
             label='per cent of birth cohort')
    ax2.set_ylabel('per cent of the birth cohort that ends up Eos-like')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5)

    # (f) apocentre
    ax = axes[5]
    for v, c, l in [(rapo[ins], C_ALL, 'all'), (rapo[eos], C_EOS, 'Eos-like'),
                    (rapo[disc], C_DISC, 'disc orbits')]:
        ax.hist(v, bins=np.linspace(0, 40, 41), density=True, histtype='step', lw=2,
                color=c, label=l)
    ax.set(xlabel=r'$r_{\rm apo}$ [kpc]', ylabel='normalised', title='(f) Apocentre')
    ax.legend(fontsize=8.5)

    # (g) integrals of motion
    ax = axes[6]
    ax.hist2d(Lz[ins] * 1e-3, E[ins] * 1e-5, bins=(140, 120),
              range=((-2.5, 6.), (-1.6, -0.2)), norm=LogNorm(), cmap='Greys')
    OT.density_contours(ax, Lz[eos] * 1e-3, E[eos] * 1e-5, [[-2.5, 6.], [-1.6, -0.2]],
                        C_EOS, label='Eos-like')
    ax.set(xlabel=r'$L_z\ [10^3$ kpc km s$^{-1}]$', ylabel=r'$E\ [10^5$ km$^2$ s$^{-2}]$',
           title='(g) Integrals of motion')
    ax.legend(fontsize=8.5, loc='lower right')

    # (h) metallicity
    ax = axes[7]
    for v, c, l in [(mh[ins], C_ALL, 'all'), (mh[eos], C_EOS, 'Eos-like'),
                    (mh[disc], C_DISC, 'disc orbits')]:
        ax.hist(v[np.isfinite(v)], bins=np.linspace(-1.6, 0.5, 50), density=True,
                histtype='step', lw=2, color=c, label=l)
    ax.set(xlabel='[M/H]', ylabel='normalised', title='(h) Metallicity')
    ax.legend(fontsize=8.5, loc='upper left')

    w = (tform > T_MERGE[0]) & (tform < T_MERGE[1])
    inside, control, enh = OT.local_enhancement(tform, eos, ins, T_MERGE)
    return dict(n_eos=int(eos.sum()), n_disc=int(disc.sum()),
                age_eos=float(np.median(age[eos])), age_disc=float(np.median(age[disc])),
                mh_eos=float(np.median(mh[eos])), mh_disc=float(np.median(mh[disc])),
                age_offset=float(off),
                eos_frac_in_merger=float(inside), eos_frac_control=float(control),
                merger_enhancement=float(enh),
                frac_eos_born_in_merger=float(w[eos].mean()))


cached = [(m, f'{G.OUT_DIR}/{m}_stars.npz') for m in ('clumpy', 'notclumpy')]
usable = [(m, np.load(p)) for m, p in cached if os.path.exists(p)]
missing = [m for m, p in cached if not os.path.exists(p)]
if missing:
    print(f'no cache for: {", ".join(missing)} (run prep_gastro.py)')
if not usable:
    raise SystemExit('nothing to plot')

fig, axes = plt.subplots(2 * len(usable), 4, figsize=(24.5, 10.6 * len(usable)), squeeze=False)
stats = {}
for k, (m, d) in enumerate(usable):
    flat = list(axes[2 * k]) + list(axes[2 * k + 1])
    stats[m] = panels(flat, d, G.LABELS[m])
    print(f'\n{G.LABELS[m]}')
    for key, val in stats[m].items():
        print(f'  {key:26s} {val}')

fig.suptitle('gastro / joaorun003: age and kinematic signature of the Eos-like population '
             f'($|v_\\phi|<{G.EOS_VPHI_MAX:.0f}$ km/s, ecc$>{G.EOS_ECC_MIN:.1f}$; '
             'total Z only, no alpha split)', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, .955])
out = G.FIG_DIR + '/gastro_eos_age_kinematics.png'
fig.savefig(out, dpi=140)
np.savez(G.OUT_DIR + '/gastro_eos_summary.npz',
         **{f'{m}_{k}': v for m, s in stats.items() for k, v in s.items()})
print('\nsaved', out)
