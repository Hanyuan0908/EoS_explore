"""Bottom-row reproduction of Borbolato et al. (2026) Figure 5, Clumpy+merger model.

Two panels:
  left   the [O/Fe]-[Fe/H] plane with the high/low-alpha boundary drawn on it,
         which is the one step of their recipe that has to be inferred rather
         than read off, so it is shown rather than asserted;
  right  their Figure 5 bottom row: the evolution of V_phi for the low- and
         high-alpha Splash against their canonical discs.

Cuts follow Borbolato et al. Section 3.1-3.2 exactly:
  * oxygen is the alpha tracer; the split is the valley in the [O/Fe] histogram
    taken over -0.7 < [Fe/H] < -0.2, with an exclusion gap either side (their
    "gap between the two populations to avoid regions where they may overlap");
  * R_GC > 5 kpc, as in their Section 3.2 ("stars at R_GC < 5 kpc are excluded
    from the analysis").  This is the literal cut, not one rescaled by the disc
    scale length: it reproduces their Splash fractions, 0.21% of the low-alpha
    and 8.10% of the high-alpha population here against 0.25% and 8.16% in their
    APOGEE sample, whereas rescaling by R_d gives 0.43% and 6.73%;
  * Splash = V_phi < 100 km/s (low-alpha) and V_phi < 50 km/s (high-alpha), the
    stricter cut for high-alpha because the simulated thick disc is more heated
    than the Milky Way's.  No eccentricity and no age cut.

Reads out/fig5_clumpy_merger.npz (built by gastro_fig5_prep.py).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gastro_config as G

os.makedirs(G.FIG_DIR, exist_ok=True)
d = np.load(G.OUT_DIR + '/fig5_clumpy_merger.npz')

# The dwarf's three pericentric passages (Borbolato et al. Sec. 2.2 and Fig. 5).
PERI = [(1.6, '1st pericentre'), (2.5, '2nd pericentre'), (3.2, 'dwarf fully disrupted')]
CLUMPY_PHASE = (0., 3.)
RMIN = 5.0
VPHI_LOW, VPHI_HIGH = 100., 50.
FEH_WINDOW = (-0.7, -0.2)
BAND = (16, 84)                      # percentile band drawn around each track
C_LOW, C_HIGH, C_DISC = '#1a9850', '#e08214', '0.15'

MODEL_DIR = G.HERE + '/jrun003.dwarfM06XY138Z37Vxy20FB20'
NAME = 'dwarfM06XY138Z37Vxy20FB20'

ofe, feh, R, vphi0, Rform = d['ofe'], d['feh'], d['R'], d['vphi'], d['Rform']
dip, gap = float(d['dip']), float(d['gap'])
times, counts = d['times'], d['counts']
zform = np.load(f'{MODEL_DIR}/{NAME}_zform.npy')

insitu = ~G.satellite_born(Rform, zform)
vol = insitu & (R > RMIN)
low = vol & (ofe < dip - gap)
high = vol & (ofe > dip + gap)
splash_low = low & (vphi0 < VPHI_LOW)
splash_high = high & (vphi0 < VPHI_HIGH)

print(f'R > {RMIN:.0f} kpc, satellite-born excluded ({(~insitu).sum():,} stars); '
      f'alpha split [O/Fe]={dip:+.3f} +/-{gap}')
print(f'  low-alpha  {low.sum():>7,} -> Splash {splash_low.sum():>6,} '
      f'({100*splash_low.sum()/low.sum():.2f}%)   [APOGEE: 0.25%]')
print(f'  high-alpha {high.sum():>7,} -> Splash {splash_high.sum():>6,} '
      f'({100*splash_high.sum()/high.sum():.2f}%)   [APOGEE: 8.16%]')


def track(mask, nmin=20):
    """Median V_phi and the percentile band, per snapshot, from the cached arrays."""
    off = np.concatenate([[0], np.cumsum(counts)])
    med = np.full(len(times), np.nan)
    lo = np.full(len(times), np.nan)
    hi = np.full(len(times), np.nan)
    for k, n in enumerate(counts):
        v = d['snap_vphi'][off[k]:off[k] + n][mask[:n]]
        if len(v) >= nmin:
            med[k] = np.median(v)
            lo[k], hi[k] = np.percentile(v, BAND)
    return med, lo, hi


fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.5, 6.8))

# ------------------------------------------------- left: the alpha boundary --
axL.hist2d(feh[vol], ofe[vol], bins=(150, 130), range=((-2.0, 0.8), (-0.45, 0.6)),
           norm=LogNorm(), cmap='Greys')
axL.axhspan(dip - gap, dip + gap, color='r', alpha=.22, lw=0)
axL.axhline(dip, color='r', lw=1.6)
for v in FEH_WINDOW:
    axL.axvline(v, color='tab:blue', lw=1.2, ls='--')
axL.text(-1.9, dip + gap + .03, 'high-$\\alpha$', color=C_HIGH, fontsize=12, va='bottom')
axL.text(-1.9, dip - gap - .03, 'low-$\\alpha$', color=C_LOW, fontsize=12, va='top')
axL.set(xlabel='[Fe/H]', ylabel='[O/Fe]',
        title=f'$\\alpha$ split at [O/Fe] $= {dip:+.2f}$ (red), $\\pm{gap}$ gap\n'
              f'dashed: the ${FEH_WINDOW[0]} <$ [Fe/H] $< {FEH_WINDOW[1]}$ window '
              'the split is measured in')

# -------------------------------------- right: Figure 5 bottom row, V_phi(t) --
axR.axvspan(*CLUMPY_PHASE, color='0.5', alpha=.18, lw=0)
for m, c, lab in [(low & ~splash_low, C_DISC, 'canonical low-$\\alpha$ disc'),
                  (high & ~splash_high, C_DISC, 'canonical high-$\\alpha$ disc'),
                  (splash_low, C_LOW, 'low-$\\alpha$ Splash'),
                  (splash_high, C_HIGH, 'high-$\\alpha$ Splash')]:
    med, lo, hi = track(m)
    ls = '--' if (c == C_DISC and 'high' in lab) else '-'
    axR.fill_between(times, lo, hi, color=c, alpha=.13, lw=0)
    axR.plot(times, med, color=c, lw=2.6, ls=ls, label=lab)

axR.axhline(VPHI_LOW, color=C_LOW, lw=1.0, ls=':')
axR.axhline(VPHI_HIGH, color=C_HIGH, lw=1.0, ls=':')
axR.text(9.85, VPHI_LOW + 4, r'$V_\phi=100$: low-$\alpha$ Splash cut', color=C_LOW,
         fontsize=8, ha='right')
axR.text(9.85, VPHI_HIGH - 13, r'$V_\phi=50$: high-$\alpha$ Splash cut', color=C_HIGH,
         fontsize=8, ha='right')

for t, lab in PERI:
    axR.axvline(t, color='k', lw=1.2)
    axR.text(t - .12, 296, lab, rotation=90, ha='right', va='top', fontsize=8.5)

axR.set(xlim=(0, 10), ylim=(-50, 300), xlabel='Time [Gyr]',
        ylabel=r'$V_\phi$ [km s$^{-1}$]')
handles, labels = axR.get_legend_handles_labels()
handles.append(Line2D([], [], color='k', lw=1.2))
labels.append('pericentric passages of the dwarf')
axR.legend(handles, labels, fontsize=8.5, ncol=3, loc='upper center',
           bbox_to_anchor=(.5, -.13), frameon=False)

fig.suptitle('Clumpy+merger (GASTRO c.r.c03): Borbolato et al. (2026) Fig. 5 bottom row, '
             f'$R_{{\\rm GC}}>{RMIN:.0f}$ kpc; shading = {BAND[0]}-{BAND[1]}th percentile',
             fontsize=12)
fig.tight_layout(rect=[0, .07, 1, .93])
out = G.FIG_DIR + '/gastro_fig5_clumpy_merger.png'
fig.savefig(out, dpi=150)

print('\n  t     splash_low   disc_low   splash_high   disc_high')
tl, _, _ = track(splash_low); dl, _, _ = track(low & ~splash_low)
th, _, _ = track(splash_high); dh, _, _ = track(high & ~splash_high)
for k in range(len(times)):
    print(f'  {times[k]:4.1f}  {tl[k]:10.1f} {dl[k]:10.1f} {th[k]:13.1f} {dh[k]:11.1f}')
print('saved', out)
