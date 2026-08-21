"""Au18: the age + kinematic signature of Eos, with the sim cut the way the data are.

The observational case for Eos being GS/E-induced star formation rests on two
measurements: Eos stars are old, and at *fixed* [Fe/H] they are older than the
rotating population of the same chemistry -- so they are not disc stars that were
later kicked onto halo orbits.  This script asks Au18 the same two questions by
applying the APOGEE/LAMOST-style orbital cuts (|v_phi| and eccentricity, from
src/eos/config.py) to the in-situ stars of the simulation, instead of selecting
on birth properties as the A/B/C channel scripts do.  The GS/E debris, measured
identically, is the accreted reference: a merger-induced in-situ population
should be old and enhanced at the merger, yet sit on clearly less extreme orbits
than the debris itself.

Note on abundances: Au18's in-situ [Fe/H] distribution is offset high relative to
the Milky Way (median +0.00 over 4<R<30 kpc), so the observational Eos window
-1.1 < [Fe/H] < -0.5 is marked for orientation but not imposed -- the analogue is
defined kinematically and the metallicity dependence is shown explicitly.

Reads out/z0_insitu_catalog.npz (built by ana_z0_kinematic_catalog.py).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import orbit_tools as OT
import config_au18 as C

os.makedirs(C.FIG_DIR, exist_ok=True)
d = np.load(C.OUT_DIR + '/z0_insitu_catalog.npz')

# Orbital operationalisation, mirroring src/eos/config.py
VPHI_MAX, ECC_MIN = 100., 0.6         # EOS_VTAN_MAX, EOS_ECC_MIN
VPHI_DISC, ECC_DISC = 150., 0.35      # DISC_VTAN_MIN, DISC_ECC_MAX
FEH_EOS = (-1.1, -0.5)                # observational Eos regime, shown not imposed
RMIN, RMAX = 4., 30.                  # analysis volume, matching the gastro figure

# Merger epoch established in PROGRESS.md (pericentre plunge -> coalescence -> end).
T_PLUNGE, T_COAL, T_END = 5.0, 5.4, 6.54
# The Eos-like birth rate spikes on the plunge itself, so the enhancement is
# measured over that narrow burst rather than the whole merger window.
T_BURST0, T_BURST1 = 4.8, 5.6
C_EOS, C_DISC, C_GSE, C_ALL = '#b2182b', '#2166ac', '#1b7837', '.55'

R, r, vphi, ecc = d['R'], d['r'], d['vphi'], d['ecc']
age, tform, feh, rapo, Lz, E = d['age'], d['tform'], d['feh'], d['rapo'], d['Lz'], d['E']
ins = (R > RMIN) & (R < RMAX) & np.isfinite(ecc) & np.isfinite(feh)
eos = ins & (np.abs(vphi) < VPHI_MAX) & (ecc > ECC_MIN)
disc = ins & (vphi > VPHI_DISC) & (ecc < ECC_DISC)
g_ok = (d['gse_R'] > RMIN) & (d['gse_R'] < RMAX) & np.isfinite(d['gse_ecc'])

print(f'in-situ, {RMIN:.0f}<R<{RMAX:.0f} kpc : {ins.sum():,}')
for lab, m, a, f_ in [('Eos-like ', eos, age[eos], feh[eos]),
                      ('disc     ', disc, age[disc], feh[disc]),
                      ('GS/E     ', g_ok, d['gse_age'][g_ok], d['gse_feh'][g_ok])]:
    print(f'  {lab} N={m.sum():>7,}   median age {np.median(a):5.2f} Gyr   '
          f'[Fe/H] {np.nanmedian(f_):+.2f}')

fig, axes = plt.subplots(2, 4, figsize=(24.5, 10.6))

# --- (a) when were the hot orbits made? --------------------------------------
ax = axes[0, 0]
ax.hist2d(tform[ins], vphi[ins], bins=(130, 120), range=((0, C.T0_GYR), (-250, 400)),
          norm=LogNorm(), cmap='Greys')
tb = np.linspace(0, C.T0_GYR, 45)
ctr = .5 * (tb[:-1] + tb[1:])
ib = np.clip(np.searchsorted(tb, tform[ins]) - 1, 0, len(tb) - 2)
vv = vphi[ins]
med = np.array([np.median(vv[ib == k]) if (ib == k).sum() > 50 else np.nan
                for k in range(len(tb) - 1)])
ax.axvspan(T_PLUNGE, T_END, color='goldenrod', alpha=.2, lw=0, label='GS/E merger')
ax.axvline(T_COAL, color='goldenrod', lw=2)
ax.plot(ctr, med, color='gold', lw=2.5, label=r'median $v_\phi$')
ax.axhline(0, color='k', lw=.6, ls=':')
ax.axhline(VPHI_MAX, color=C_EOS, lw=.9, ls='--')
ax.axhline(-VPHI_MAX, color=C_EOS, lw=.9, ls='--')
ax.set(xlabel='birth cosmic time [Gyr]', ylabel=r'$v_\phi$ [km s$^{-1}$]',
       title='(a) Disc spin-up')
ax.legend(fontsize=8.5, loc='lower right')

# --- (b) where the selections sit in the orbit plane --------------------------
ax = axes[0, 1]
ax.hist2d(vphi[ins], ecc[ins], bins=(120, 100), range=((-250, 400), (0, 1)),
          norm=LogNorm(), cmap='Greys')
ax.add_patch(Rectangle((-VPHI_MAX, ECC_MIN), 2 * VPHI_MAX, 1 - ECC_MIN,
                       fill=False, ec=C_EOS, lw=2, label='Eos-like'))
ax.add_patch(Rectangle((VPHI_DISC, 0), 400 - VPHI_DISC, ECC_DISC,
                       fill=False, ec=C_DISC, lw=2, label='disc orbits'))
ax.plot(np.median(d['gse_vphi'][g_ok]), np.nanmedian(d['gse_ecc'][g_ok]), '*',
        color=C_GSE, ms=18, mec='k', mew=.6, label='GS/E debris (median)')
ax.set(xlabel=r'$v_\phi$ [km s$^{-1}$]', ylabel='eccentricity',
       title='(b) Selection in the orbit plane')
ax.legend(fontsize=8.5, loc='lower left')

# --- (c) ages of the two orbital populations ----------------------------------
ax = axes[0, 2]
bins = np.linspace(0, C.T0_GYR, 45)
ax.hist(age[ins], bins=bins, density=True, histtype='stepfilled', color=C_ALL, alpha=.3,
        label=f'all in-situ (N={ins.sum():,})')
for m, c, l in [(eos, C_EOS, f'Eos-like (N={eos.sum():,})'),
                (disc, C_DISC, f'disc orbits (N={disc.sum():,})')]:
    ax.hist(age[m], bins=bins, density=True, histtype='step', lw=2, color=c, label=l)
    ax.axvline(np.median(age[m]), color=c, ls=':', lw=1.4)
ax.hist(d['gse_age'][g_ok], bins=bins, density=True, histtype='step', lw=1.8, ls='--',
        color=C_GSE, label=f'GS/E debris (N={g_ok.sum():,})')
ax.axvspan(C.T0_GYR - T_END, C.T0_GYR - T_PLUNGE, color='goldenrod', alpha=.2, lw=0)
ax.set(xlabel='age [Gyr]', ylabel='normalised density', title='(c) Age by orbit type')
ax.legend(fontsize=8)

# --- (d) the decisive observational test --------------------------------------
ax = axes[0, 3]
edges = np.linspace(-1.4, 0.4, 19)
cen = .5 * (edges[:-1] + edges[1:])


def track(mask):
    med, lo, hi, n = [], [], [], []
    for i in range(len(cen)):
        a = age[mask & (feh >= edges[i]) & (feh < edges[i + 1])]
        if len(a) >= 30:
            med.append(np.median(a)); lo.append(np.percentile(a, 25)); hi.append(np.percentile(a, 75))
        else:
            med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
        n.append(len(a))
    return np.array(med), np.array(lo), np.array(hi), np.array(n)


hm, hlo, hhi, hn = track(eos)
dm, dlo, dhi, dn = track(disc)
ax.axvspan(*FEH_EOS, color='k', alpha=.07, lw=0)
ax.fill_between(cen, hlo, hhi, color=C_EOS, alpha=.15)
ax.fill_between(cen, dlo, dhi, color=C_DISC, alpha=.15)
ax.plot(cen, hm, 's-', color=C_EOS, lw=2, ms=5, label='Eos-like (hot, eccentric)')
ax.plot(cen, dm, 'o-', color=C_DISC, lw=2, ms=5, label='disc orbits')
ax.set(xlabel='[Fe/H]', ylabel='median age [Gyr]',
       title='(d) Age at fixed [Fe/H] (the decisive test)')
ax.text(np.mean(FEH_EOS), ax.get_ylim()[0] + .3, 'observed\nEos regime', ha='center',
        fontsize=7.5, color='.3')
ax.legend(fontsize=8.5, loc='upper right')
both = np.isfinite(hm) & np.isfinite(dm)
eos_reg = both & (cen > FEH_EOS[0]) & (cen < FEH_EOS[1])
off_all, off_eos = np.mean((hm - dm)[both]), np.mean((hm - dm)[eos_reg])
ax.text(.03, .04, f'mean offset: {off_all:+.2f} Gyr overall,\n'
                  f'{off_eos:+.2f} Gyr in the Eos regime',
        transform=ax.transAxes, fontsize=8.5, bbox=dict(fc='white', alpha=.85, ec='none'))

# --- (e) is the Eos-like population made by the merger? -----------------------
ax = axes[1, 0]
n_eos, _ = np.histogram(tform[eos], bins=tb)
ax.fill_between(ctr, 0, n_eos, step='mid', color=C_EOS, alpha=.35, lw=0)
ax.step(ctr, n_eos, where='mid', color=C_EOS, lw=2, label='Eos-like births per bin')
ax.axvspan(T_PLUNGE, T_END, color='goldenrod', alpha=.2, lw=0)
ax.axvline(T_COAL, color='goldenrod', lw=2)
ax.set(xlabel='birth cosmic time [Gyr]', ylabel='Eos-like stars per bin',
       title='(e) Formation history of the Eos analogue')
_in, _ctrl, _enh = OT.local_enhancement(tform, eos, ins, (T_BURST0, T_BURST1))
ax.text(.97, .95, f'burst {T_BURST0}-{T_BURST1} Gyr:\n{100*_in:.0f}% of the cohort ends up\n'
                  f'Eos-like vs {100*_ctrl:.0f}% either side (x{_enh:.1f})',
        transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
        bbox=dict(fc='white', alpha=.85, ec='none'))
ax2 = ax.twinx()
ntot, _ = np.histogram(tform[ins], bins=tb)
ok = ntot > 200
ax2.plot(ctr[ok], 100 * n_eos[ok] / ntot[ok], color='k', lw=1.6, ls='--',
         label='per cent of birth cohort')
ax2.set_ylabel('per cent of the birth cohort that ends up Eos-like')
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8.5)

# --- (f) apocentre ------------------------------------------------------------
ax = axes[1, 1]
b = np.linspace(0, 40, 41)
for v, c, l, ls in [(rapo[ins], C_ALL, 'all in-situ', '-'),
                    (rapo[eos], C_EOS, 'Eos-like', '-'),
                    (rapo[disc], C_DISC, 'disc orbits', '-'),
                    (d['gse_rapo'][g_ok], C_GSE, 'GS/E debris', '--')]:
    ax.hist(v, bins=b, density=True, histtype='step', lw=2, color=c, ls=ls, label=l)
ax.set(xlabel=r'$r_{\rm apo}$ [kpc]', ylabel='normalised', title='(f) Apocentre')
ax.legend(fontsize=8.5)

# --- (g) angular momentum vs energy -------------------------------------------
ax = axes[1, 2]
ax.hist2d(Lz[ins] * 1e-3, E[ins] * 1e-5, bins=(140, 120),
          range=((-1.5, 3.5), (-2.2, -0.9)), norm=LogNorm(), cmap='Greys')
LZE_RNG = [[-1.5, 3.5], [-2.2, -0.9]]
for xx, yy, c, l, ls in [(Lz[eos], E[eos], C_EOS, 'Eos-like', '-'),
                         (d['gse_Lz'][g_ok], d['gse_E'][g_ok], C_GSE, 'GS/E debris', '--')]:
    OT.density_contours(ax, xx * 1e-3, yy * 1e-5, LZE_RNG, c, label=l, ls=ls)
ax.set(xlabel=r'$L_z\ [10^3$ kpc km s$^{-1}]$', ylabel=r'$E\ [10^5$ km$^2$ s$^{-2}]$',
       title='(g) Integrals of motion')
ax.legend(fontsize=8.5, markerscale=4, loc='lower right')

# --- (h) metallicity ----------------------------------------------------------
ax = axes[1, 3]
b = np.linspace(-2.5, 0.7, 50)
ax.axvspan(*FEH_EOS, color='k', alpha=.07, lw=0)
for v, c, l, ls in [(feh[ins], C_ALL, 'all in-situ', '-'),
                    (feh[eos], C_EOS, 'Eos-like', '-'),
                    (feh[disc], C_DISC, 'disc orbits', '-'),
                    (d['gse_feh'][g_ok], C_GSE, 'GS/E debris', '--')]:
    ax.hist(v[np.isfinite(v)], bins=b, density=True, histtype='step', lw=2, color=c, ls=ls, label=l)
ax.set(xlabel='[Fe/H]', ylabel='normalised', title='(h) Metallicity')
ax.legend(fontsize=8.5, loc='upper left')

fig.suptitle('Au18: age and kinematic signature of the Eos analogue, selected as in the data '
             f'(in-situ, {RMIN:.0f}<R<{RMAX:.0f} kpc, '
             f'$|v_\\phi|<{VPHI_MAX:.0f}$ km/s, ecc$>{ECC_MIN:.1f}$)', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, .955])
out = C.FIG_DIR + '/au18_eos_age_kinematics.png'
fig.savefig(out, dpi=140)

# ------------------------------------------------------------------- numbers --
print('\n[Fe/H]     age(Eos-like)  age(disc)   offset   N_eos  N_disc')
for i in range(len(cen)):
    if np.isfinite(hm[i]) or np.isfinite(dm[i]):
        print(f'  {cen[i]:+.2f}      {hm[i]:6.2f}       {dm[i]:6.2f}   {hm[i]-dm[i]:+6.2f}'
              f'   {hn[i]:6d} {dn[i]:7d}')
print(f'\nmean age offset (Eos-like - disc): {off_all:+.2f} Gyr overall, '
      f'{off_eos:+.2f} Gyr over the observed Eos regime {FEH_EOS}')
inside, control, enh = OT.local_enhancement(tform, eos, ins, (T_BURST0, T_BURST1))
print(f'\nburst window {T_BURST0}-{T_BURST1} Gyr: {100*inside:.1f}% of that birth cohort ends up '
      f'Eos-like, vs {100*control:.1f}% in the flanking control intervals '
      f'(enhancement x{enh:.2f})')
w = (tform > T_PLUNGE) & (tform < T_END)
print(f'whole merger window {T_PLUNGE}-{T_END} Gyr: {100*w[ins].mean():.1f}% of all in-situ '
      f'births, {100*w[eos].mean():.1f}% of Eos-like births')
print(f'Eos-like fraction of the birth cohort: '
      f'{100*np.mean(eos[ins & (tform < T_PLUNGE)]):.1f}% pre-merger, '
      f'{100*np.mean(eos[ins & w]):.1f}% during, '
      f'{100*np.mean(eos[ins & (tform > T_END)]):.1f}% after')
print(f'median r_apo: Eos-like {np.median(rapo[eos]):.1f}, disc {np.median(rapo[disc]):.1f}, '
      f'GS/E {np.nanmedian(d["gse_rapo"][g_ok]):.1f} kpc')
print('saved', out)
