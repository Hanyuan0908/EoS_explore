"""Bottom-row reproduction of Borbolato et al. (2026) Figure 5, Clumpy+merger model.

Two panels:
  left   the [O/Fe]-[Fe/H] plane with the selection drawn on it, so the cuts are
         visible rather than asserted;
  right  their Figure 5 bottom row: the evolution of V_phi for the low- and
         high-alpha Splash against their canonical discs.

Every cut is taken from the paper rather than inferred:

  * alpha tracer is oxygen.  The high/low boundaries are the values printed in
    their Figure 3, column 4 (Clumpy+merger): **low-alpha [O/Fe] < -0.13,
    high-alpha [O/Fe] > +0.10**, with the span between them discarded -- their
    "gap between the two populations to avoid regions where they may overlap".
    Measuring the valley of the [O/Fe] histogram over -0.7 < [Fe/H] < -0.2
    independently gives -0.155, consistent with their -0.13.
  * **[Fe/H] > -1.0** (their Section 3.1, "we select low- and high-alpha disks
    using [Mg/Fe]-[Fe/H] for stars with [Fe/H] > -1.0").  This matters far more
    than it looks: without it the low-alpha sample picks up a metal-poor tail
    ([Fe/H] ~ -1.2) of stars that are chemically odd rather than disc stars --
    at t_form < 1 Gyr they are 1.7% of their birth cohort, sitting at
    [O/Fe] = -0.29 while the cohort sits at +0.28.  They are already dynamically
    hot, and they drag the early Splash track down by ~25 km/s.
  * R_GC > 5 kpc (their Section 3.2), taken literally rather than rescaled by the
    disc scale length: it reproduces their Splash fractions.
  * Splash = V_phi < 100 km/s (low-alpha), V_phi < 50 km/s (high-alpha), the
    stricter cut for high-alpha because the simulated thick disc is more heated
    than the Milky Way's.  No eccentricity cut.
  * **t_form < 4 Gyr**, the low-alpha sample definition printed on their Figure 3.
    It only touches the canonical disc -- every Splash star here is older than
    that already -- and it is what brings the disc track to 241 km/s at t = 10 Gyr
    against the 240 read off their figure (264 without it).

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

# --- the paper's cuts --------------------------------------------------------
OFE_LOW, OFE_HIGH = -0.13, 0.10      # Borbolato et al. Fig. 3, col. 4
FEH_MIN = -1.0                       # their Sec. 3.1
RMIN = 5.0                           # their Sec. 3.2
VPHI_LOW, VPHI_HIGH = 100., 50.      # Splash cuts
FEH_WINDOW = (-0.7, -0.2)            # window the split is measured in
TFORM_MAX = 4.0                      # their Fig. 3 low-alpha sample; affects the disc only
NMIN = 15                            # a track point needs at least this many stars
BAND = (16, 84)
# The dwarf's three pericentric passages (their Sec. 2.2 and Fig. 5).
PERI = [(1.6, '1st pericentre'), (2.5, '2nd pericentre'), (3.2, 'dwarf fully disrupted')]
CLUMPY_PHASE = (0., 3.)
C_LOW, C_HIGH, C_DISC = '#1a9850', '#e08214', '0.15'

MODEL_DIR = G.HERE + '/jrun003.dwarfM06XY138Z37Vxy20FB20'
NAME = 'dwarfM06XY138Z37Vxy20FB20'

ofe, feh, R, vphi0, Rform = d['ofe'], d['feh'], d['R'], d['vphi'], d['Rform']
times, counts = d['times'], d['counts']
zform = np.load(f'{MODEL_DIR}/{NAME}_zform.npy')

insitu = ~G.satellite_born(Rform, zform)
vol = insitu & (R > RMIN) & (feh > FEH_MIN) & (d['tform'] < TFORM_MAX)
low = vol & (ofe < OFE_LOW)
high = vol & (ofe > OFE_HIGH)
splash_low = low & (vphi0 < VPHI_LOW)
splash_high = high & (vphi0 < VPHI_HIGH)

print(f'R > {RMIN:.0f} kpc, [Fe/H] > {FEH_MIN}, t_form < {TFORM_MAX} Gyr, '
      f'satellite-born excluded ({(~insitu).sum():,})')
# Splash fractions are quoted against the full-age parent population: the
# t_form cut shapes the disc track, it is not part of defining the Splash.
allage = insitu & (R > RMIN) & (feh > FEH_MIN)
for lab, sel, sp, ref in [('low-alpha ', allage & (ofe < OFE_LOW), splash_low, 0.25),
                          ('high-alpha', allage & (ofe > OFE_HIGH), splash_high, 8.16)]:
    print(f'  {lab} {sel.sum():>7,} -> Splash {sp.sum():>6,} '
          f'({100*sp.sum()/sel.sum():.2f}%)   [their APOGEE: {ref}%]')


def track(mask):
    """Median V_phi and percentile band per snapshot, from the cached arrays.

    Stars that have not formed yet simply are not in that snapshot's array, so
    the early points are built from whatever exists -- the track starts where the
    population first has NMIN members, which is t = 1 Gyr, as in their figure.
    """
    off = np.concatenate([[0], np.cumsum(counts)])
    med, lo, hi = (np.full(len(times), np.nan) for _ in range(3))
    for k, n in enumerate(counts):
        v = d['snap_vphi'][off[k]:off[k] + n][mask[:n]]
        if len(v) >= NMIN:
            med[k] = np.median(v)
            lo[k], hi[k] = np.percentile(v, BAND)
    return med, lo, hi


fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.5, 6.8))

# ------------------------------------------------- left: the selection plane --
sel_vol = insitu & (R > RMIN)
axL.hist2d(feh[sel_vol], ofe[sel_vol], bins=(150, 130), range=((-2.0, 0.8), (-0.45, 0.6)),
           norm=LogNorm(), cmap='Greys')
axL.axhspan(OFE_LOW, OFE_HIGH, color='r', alpha=.13, lw=0)
for y in (OFE_LOW, OFE_HIGH):
    axL.axhline(y, color='r', lw=1.5, ls='--')
for x in FEH_WINDOW:
    axL.axvline(x, color='k', lw=1.1)
axL.axvline(FEH_MIN, color='tab:blue', lw=1.8, ls=':')
axL.text(-1.97, OFE_HIGH + .03, f'high-$\\alpha$: [O/Fe] $> {OFE_HIGH:+.2f}$',
         color=C_HIGH, fontsize=10, va='bottom')
axL.text(-1.97, OFE_LOW - .03, f'low-$\\alpha$: [O/Fe] $< {OFE_LOW:+.2f}$',
         color=C_LOW, fontsize=10, va='top')
axL.text(FEH_MIN + .03, .55, '[Fe/H] $>-1.0$', color='tab:blue', fontsize=9, rotation=90,
         va='top')
axL.set(xlabel='[Fe/H]', ylabel='[O/Fe]',
        title='Selection: $\\alpha$ boundaries from their Fig. 3 (dashed, gap shaded)\n'
              'solid black = the $-0.7<$ [Fe/H] $<-0.2$ window; dotted = the [Fe/H] floor')
axL.text(.98, .03, f'also: $R_{{\\rm GC}}>{RMIN:.0f}$ kpc, $t_{{\\rm form}}<{TFORM_MAX:.0f}$ Gyr',
         transform=axL.transAxes, ha='right', fontsize=8.5, color='.25')

# -------------------------------------- right: Figure 5 bottom row, V_phi(t) --
axR.axvspan(*CLUMPY_PHASE, color='0.5', alpha=.18, lw=0)
for m, c, lab, ls in [(low & ~splash_low, C_DISC, 'canonical low-$\\alpha$ disc', '-'),
                      (high & ~splash_high, C_DISC, 'canonical high-$\\alpha$ disc', '--'),
                      (splash_low, C_LOW, 'low-$\\alpha$ Splash', '-'),
                      (splash_high, C_HIGH, 'high-$\\alpha$ Splash', '-')]:
    med, lo, hi = track(m)
    axR.fill_between(times, lo, hi, color=c, alpha=.13, lw=0)
    axR.plot(times, med, color=c, lw=2.6, ls=ls, label=lab)

axR.axhline(VPHI_LOW, color=C_LOW, lw=1.0, ls=':')
axR.axhline(VPHI_HIGH, color=C_HIGH, lw=1.0, ls=':')
axR.text(9.85, VPHI_LOW + 5, r'$V_\phi=100$: low-$\alpha$ Splash cut', color=C_LOW,
         fontsize=8, ha='right')
axR.text(9.85, VPHI_HIGH - 14, r'$V_\phi=50$: high-$\alpha$ Splash cut', color=C_HIGH,
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

fig.suptitle('Clumpy+merger (GASTRO c.r.c03): Borbolato et al. (2026) Fig. 5 bottom row  '
             f'[in-situ, $R_{{\\rm GC}}>{RMIN:.0f}$ kpc, [Fe/H]$>{FEH_MIN}$, $t_{{\\rm form}}<{TFORM_MAX:.0f}$ Gyr; '
             f'shading = {BAND[0]}-{BAND[1]}th percentile]', fontsize=12)
fig.tight_layout(rect=[0, .07, 1, .93])
out = G.FIG_DIR + '/gastro_fig5_clumpy_merger.png'
fig.savefig(out, dpi=150)

gl, _, _ = track(splash_low); dl, _, _ = track(low & ~splash_low)
gh, _, _ = track(splash_high); dh, _, _ = track(high & ~splash_high)
print('\n  t    low-a Splash  low-a disc | high-a Splash  high-a disc'
      '     (their Fig. 5, read off: green 185 at t=1, black 182)')
for k in range(len(times)):
    print(f'  {times[k]:4.1f}  {gl[k]:12.1f} {dl[k]:11.1f} | {gh[k]:13.1f} {dh[k]:12.1f}')
print('saved', out)
