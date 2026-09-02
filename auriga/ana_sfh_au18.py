"""Star-formation history of Au18, and where the GS/E merger falls in it.

The SFR is built from GFM_InitialMass and GFM_StellarFormationTime of the in-situ
stars inside 0.15 R200 at z=0 (the sample assembled by date_merger_z0.py), so it
is the formation history of the surviving main galaxy, not of the whole box.  It
uses INITIAL mass: present-day mass has already lost ~30 per cent to stellar
winds and SNe, and that loss is age-dependent, which would tilt the old end of
the history downwards if current masses were used.

Because the formation time is stored per particle, the SFR can be binned as
finely as particle numbers allow -- it is not limited by the ~0.15 Gyr snapshot
spacing that sets the resolution of the merger track in the lower-left panel.

The GS/E timeline comes from date_merger_track.py (out/gse_track_clean_*.npz):
the debris sits at r~210 kpc until first apocentre, plunges in, and phase-mixes
at t = 5.3-5.6 Gyr.  The question this figure answers is whether the in-situ
starburst is coincident with that coalescence.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import config_au18 as C

os.makedirs(C.FIG_DIR, exist_ok=True)

DT = 0.1                                  # SFR bin width [Gyr]
T_APO = 3.25                              # first apocentre of the GS/E debris
T_PERI = 5.0                              # pericentre plunge
T_COAL_LO, T_COAL_HI = 5.3, 5.6           # coalescence / phase-mixing
T_WIN = (4.99, 6.54)                      # merger window used to select Eos stars
cH, cG, cM = '#2166ac', '#b2182b', 'goldenrod'

d = np.load(C.OUT_DIR + '/z0_dating.npz')
t, mi = d['insitu_birth_age'], d['insitu_imass']         # cosmic time [Gyr], Msun
g = np.load(C.OUT_DIR + '/gse_clean_z0.npz')
tg, mg = g['age'], g['im']                               # GS/E progenitor's own stars
trk = np.load(C.OUT_DIR + '/gse_track_clean_55_127_2.npz')

bins = np.arange(0., C.T0_GYR + DT, DT)
ctr = .5 * (bins[:-1] + bins[1:])
sfr = np.histogram(t, bins=bins, weights=mi)[0] / (DT * 1e9)      # Msun/yr
sfr_g = np.histogram(tg, bins=bins, weights=mg)[0] / (DT * 1e9)


def smooth(y, sig_gyr=0.25):
    """Gaussian smooth in time, normalised so the edges are not pulled to zero."""
    x = np.arange(-4 * sig_gyr, 4 * sig_gyr + DT, DT)
    k = np.exp(-.5 * (x / sig_gyr) ** 2)
    return np.convolve(y, k, 'same') / np.convolve(np.ones_like(y), k, 'same')


# --- burst amplitude: the peak against the SFR either side of the merger ------
pk = np.argmax(smooth(sfr))
quiet = ((ctr > 3.0) & (ctr < 4.5)) | ((ctr > 7.0) & (ctr < 9.0))
base = np.median(sfr[quiet])
win = (t >= T_WIN[0]) & (t <= T_WIN[1])

print(f'in-situ, r < 0.15 R200:  N = {len(t):,}   M_formed = {mi.sum():.3e} Msun')
print(f'peak SFR              {smooth(sfr)[pk]:6.2f} Msun/yr at t = {ctr[pk]:.2f} Gyr '
      f'(lookback {C.T0_GYR - ctr[pk]:.2f} Gyr)')
print(f'baseline SFR          {base:6.2f} Msun/yr (3.0-4.5 and 7.0-9.0 Gyr)')
print(f'burst amplitude       {smooth(sfr)[pk] / base:6.2f}x baseline')
print(f'coalescence           t = {T_COAL_LO}-{T_COAL_HI} Gyr; peak is '
      f'{ctr[pk] - .5 * (T_COAL_LO + T_COAL_HI):+.2f} Gyr from its centre')
print(f'mass formed in window {T_WIN[0]}-{T_WIN[1]} Gyr: {mi[win].sum():.3e} Msun '
      f'= {100 * mi[win].sum() / mi.sum():.1f} per cent of the in-situ mass')
for f in (10, 50, 90):
    o = np.argsort(t); cm = np.cumsum(mi[o]) / mi.sum()
    print(f'  t({f:>2d} per cent of mass formed) = {t[o][np.searchsorted(cm, f / 100)]:5.2f} Gyr')
print(f'GS/E progenitor: M = {mg.sum():.3e} Msun, its own SF stops at '
      f't = {np.percentile(tg, 90):.2f} Gyr (90th pct)')

# ------------------------------------------------------------------ figure --
fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8),
                         gridspec_kw=dict(height_ratios=[1.25, 1]))


def mark_merger(ax, label=False):
    ax.axvspan(*T_WIN, color=cM, alpha=.10, lw=0,
               label='Eos selection window' if label else None)
    ax.axvspan(T_COAL_LO, T_COAL_HI, color=cM, alpha=.45, lw=0,
               label='GS/E coalescence' if label else None)
    ax.axvline(T_PERI, color=cM, ls='--', lw=1.6,
               label='pericentre plunge' if label else None)
    ax.axvline(T_APO, color=cM, ls=':', lw=1.6,
               label='first apocentre' if label else None)


def z_axis(ax):
    """Top axis in redshift, interpolated off a dense grid (t and z run opposite)."""
    zg = np.concatenate([np.linspace(20, 0.001, 400), [0.]])
    tg_ = C.COSMO.age(zg).value
    o = np.argsort(tg_)
    sec = ax.secondary_xaxis('top', functions=(
        lambda x: np.interp(x, tg_[o], zg[o]),
        lambda x: np.interp(x, zg[::-1], tg_[::-1])))
    sec.set_xticks([5, 3, 2, 1.5, 1, .5, .2, 0])
    sec.set_xlabel('redshift')


# (a) the star-formation history itself
ax = axes[0, 0]
mark_merger(ax, label=True)
ax.fill_between(ctr, sfr, step='mid', color=cH, alpha=.25, lw=0)
ax.step(ctr, sfr, where='mid', color=cH, lw=.9, alpha=.75)
ax.plot(ctr, smooth(sfr), color=cH, lw=2.4, label=f'in-situ ({DT * 1000:.0f} Myr bins, smoothed)')
ax.axhline(base, color='.35', ls='-.', lw=1.2, label=f'baseline {base:.1f} M$_\\odot$/yr')
ax.plot(ctr[pk], smooth(sfr)[pk], 'o', color='k', ms=6, zorder=5)
ax.annotate(f'peak {smooth(sfr)[pk]:.1f} M$_\\odot$/yr\nt = {ctr[pk]:.2f} Gyr',
            (ctr[pk], smooth(sfr)[pk]), textcoords='offset points', xytext=(-112, -4),
            fontsize=9, color='k')
axg = ax.twinx()
axg.plot(ctr, sfr_g, color=cG, lw=1.5, alpha=.85)
axg.set_ylabel('GS/E progenitor SFR [M$_\\odot$/yr]', color=cG, fontsize=9)
axg.tick_params(axis='y', labelcolor=cG, labelsize=8)
axg.set_ylim(0, max(sfr_g) * 3.2)
ax.set(xlim=(0, C.T0_GYR), ylim=(0, 1.15 * sfr.max()),
       ylabel='SFR [M$_\\odot$ yr$^{-1}$]',
       title='(a) Au18 star-formation history (in-situ, r < 0.15 R$_{200}$)')
ax.legend(fontsize=8.5, loc='upper right', framealpha=.9)
z_axis(ax)

# (b) the age distribution, which is what an observer measures
ax = axes[0, 1]
abins = C.T0_GYR - bins[::-1]
ax.hist(C.T0_GYR - t, bins=abins, weights=mi, density=True, histtype='stepfilled',
        color=cH, alpha=.28, lw=0, label='mass-weighted')
ax.hist(C.T0_GYR - t, bins=abins, density=True, histtype='step', lw=2,
        color='k', label='number-weighted')
ax.axvspan(C.T0_GYR - T_WIN[1], C.T0_GYR - T_WIN[0], color=cM, alpha=.10, lw=0)
ax.axvspan(C.T0_GYR - T_COAL_HI, C.T0_GYR - T_COAL_LO, color=cM, alpha=.45, lw=0)
ax.set(xlim=(0, C.T0_GYR), xlabel='age [Gyr]', ylabel='normalised density',
       title='(b) Stellar age distribution')
ax.invert_xaxis()
ax.legend(fontsize=8.5, loc='upper left')

# (c) the merger clock: the debris falls in and phase-mixes
ax = axes[1, 0]
mark_merger(ax)
ax.fill_between(trk['times'], trk['r_p25'], trk['r_p75'], color=cG, alpha=.22, lw=0)
ax.plot(trk['times'], trk['r_med'], color=cG, lw=2.2, marker='o', ms=3,
        label='GS/E debris, median r')
axd = ax.twinx()
ld, = axd.plot(trk['times'], trk['disp'], color='.35', lw=1.4, ls='--',
               label='clump dispersion (right axis)')
axd.set_ylabel('clump dispersion [kpc]', color='.35', fontsize=9)
axd.tick_params(axis='y', labelcolor='.35', labelsize=8)
axd.set_yscale('log')
ax.set(xlim=(0, C.T0_GYR), yscale='log',
       xlabel='cosmic time [Gyr]', ylabel='galactocentric radius [kpc]',
       title='(c) GS/E orbital decay and phase-mixing')
ax.legend(handles=ax.get_legend_handles_labels()[0] + [ld],
          fontsize=8.5, loc='lower left')

# (d) how much of the galaxy was already in place when the merger happened
ax = axes[1, 1]
mark_merger(ax)
o = np.argsort(t)
cum = np.cumsum(mi[o]) / mi.sum()
ax.plot(t[o], 100 * cum, color=cH, lw=2.2)
for tt, ls in [(T_PERI, '--'), (.5 * (T_COAL_LO + T_COAL_HI), '-')]:
    f = 100 * np.interp(tt, t[o], cum)
    ax.plot([0, tt], [f, f], color='.35', lw=.9, ls=ls)
    ax.annotate(f'{f:.0f}%', (0.3, f), fontsize=9, color='.25', va='bottom')
ax.set(xlim=(0, C.T0_GYR), ylim=(0, 100), xlabel='cosmic time [Gyr]',
       ylabel='per cent of in-situ mass formed',
       title='(d) Cumulative in-situ mass')

fig.suptitle('Au18: the in-situ starburst coincides with GS/E coalescence', y=.985)
fig.tight_layout(rect=[0, 0, 1, .945])
out = C.FIG_DIR + '/au18_sfh_vs_gse_merger.png'
fig.savefig(out, dpi=150)
np.savez(C.OUT_DIR + '/sfh_au18.npz', t_bin=ctr, sfr=sfr, sfr_gse=sfr_g,
         sfr_smooth=smooth(sfr), baseline=base, t_peak=ctr[pk],
         trk_times=trk['times'], trk_r_med=trk['r_med'], trk_disp=trk['disp'])
print('\nsaved', out)
