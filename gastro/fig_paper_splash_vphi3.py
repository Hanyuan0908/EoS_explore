"""PUBLICATION FIGURE: the Splash selection and its V_phi evolution, three panels.

The three-panel version of `fig_paper_splash_vphi.py`; that two-panel figure is
kept as it is, this one adds the selection panel that was previously implicit.

  (a) the [O/Fe]-[Fe/H] plane with the alpha boundaries drawn on it
  (b) the z=0 V_phi distribution of the two alpha populations, with the Splash
      cuts that make panel (c) drawn on it
  (c) median V_phi against time for the two Splash populations and their
      canonical discs, with the dwarf's three pericentric passages marked and
      the observed Eos rotation drawn as a reference line

Panel (b) is what turns (a) into (c): the alpha split in (a) gives the two
populations, the vertical lines in (b) cut each one into a Splash and a canonical
disc, and (c) follows those four sets back in time.  Both histograms are
normalised to unit area, so the panel compares the *shapes*: at fixed [O/Fe]
selection the high-alpha population carries a far heavier slow-rotating tail
(18.1 per cent of it inside |V_phi| < 80 km/s against 1.1 per cent of the
low-alpha), which is the Splash.

Clumpy+merger (GASTRO c.r.c03), after Borbolato et al. (2026) Figure 5 bottom row,
but with a single symmetric Splash window applied to both alpha populations --
|V_phi| < 80 km/s, this project's observational mask (SPLASH_VTAN_MAX in
../src/eos/config.py) -- so that low- and high-alpha are selected like for like.
Run with `paper` to use their asymmetric cuts (V_phi < 100 and < 50) instead; the
two lines in (b) then sit at different velocities and are drawn in the colour of
the population each applies to.

Writes PNG and PDF at 300 dpi with a tight bounding box into ../Fig_paper/.
Everything that carries data is rasterised; axes, text and lines stay vector.

Reads out/fig5_clumpy_merger.npz (built by gastro_fig5_prep.py).
Run with the pynbody environment (gastro_config imports pynbody):
/data/ioasoft/software/miniforge3/envs/python-3.11-2026-01a/bin/python3
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gastro_config as G

FIG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Fig_paper'
os.makedirs(FIG_DIR, exist_ok=True)

# --- publication style -------------------------------------------------------
# Times New Roman if the system has it.  Liberation Serif is preferred as the
# fallback: it is metric-compatible with Times New Roman and is a genuine
# TrueType face, so it embeds cleanly under 'pdf.fonttype': 42 (Nimbus Roman is
# also Times-metric but is OpenType/CFF, which makes PDF preflight tools warn).
# STIX is the Times-matched maths font.
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'Nimbus Roman',
                   'STIX Two Text', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 13, 'axes.labelsize': 15, 'legend.fontsize': 11.5,
    'xtick.labelsize': 12.5, 'ytick.labelsize': 12.5,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'axes.linewidth': 1.0, 'savefig.bbox': 'tight',
    # Type 42 (TrueType) rather than matplotlib's default Type 3, which many
    # journals reject at submission.
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

d = np.load(G.OUT_DIR + '/fig5_clumpy_merger.npz')

# --- cuts (see ana_gastro_fig5.py and ../gastro/README.md for provenance) -----
OFE_LOW, OFE_HIGH = -0.13, 0.10      # Borbolato et al. Fig. 3, col. 4
FEH_MIN = -1.0                       # their Sec. 3.1
RMIN = 5.0                           # their Sec. 3.2
TFORM_MAX = 4.0                      # their Fig. 3 low-alpha sample; disc only
# The alpha boundaries below were located from the [O/Fe] histogram taken over
# -0.7 < [Fe/H] < -0.2, but that window plays no part in the selection: it is
# applied across the full metallicity range, so it is not drawn on the panel.
NMIN, BAND, NBOOT = 15, (16, 84), 500
# The bootstrap error on the median is 0.7-5.2 km/s, invisible on a 475 km/s
# axis, so every band in panel (c) -- the four tracks and the observed Eos line
# alike -- is inflated by this factor about its central value.  They are then
# magnified error bars, not intervals anything falls in, and nothing on the
# figure says so: THE CAPTION MUST STATE THE FACTOR.  Set to 1 for true widths.
ERR_SCALE = 5
RNG = np.random.default_rng(0)       # fixed seed: the band is reproducible
PERI = [(1.6, '1st pericentre'), (2.5, '2nd pericentre'), (3.2, 'dwarf disrupted')]
C_LOW, C_HIGH, C_DISC = '#1a9850', '#e08214', '0.15'
# The observed Eos rotation, +4.6 km/s.  An external number, not measured from
# this simulation or from results/lamost_eos_sample.fits -- it is drawn as a
# reference for the simulated tracks, so quote its source in the caption.
V_EOS, V_EOS_ERR, C_EOS = 4.6, 3.0, '#E8112D'
VRNG = (-150., 400.)                 # covers both populations to <0.2 per cent

MODE = 'paper' if 'paper' in sys.argv[1:] else 'symmetric'
if MODE == 'paper':
    VPHI_LOW, VPHI_HIGH, SUFFIX = 100., 50., '_papercuts'
    SPLASH = lambda v, lim: v < lim
else:
    VPHI_LOW = VPHI_HIGH = 80.
    SUFFIX = ''
    SPLASH = lambda v, lim: np.abs(v) < lim

MODEL_DIR = G.HERE + '/jrun003.dwarfM06XY138Z37Vxy20FB20'
NAME = 'dwarfM06XY138Z37Vxy20FB20'
ofe, feh, R, vphi0 = d['ofe'], d['feh'], d['R'], d['vphi']
times, counts = d['times'], d['counts']
zform = np.load(f'{MODEL_DIR}/{NAME}_zform.npy')

insitu = ~G.satellite_born(d['Rform'], zform)
vol = insitu & (R > RMIN) & (feh > FEH_MIN) & (d['tform'] < TFORM_MAX)
low, high = vol & (ofe < OFE_LOW), vol & (ofe > OFE_HIGH)
splash_low = low & SPLASH(vphi0, VPHI_LOW)
splash_high = high & SPLASH(vphi0, VPHI_HIGH)
print(f'[{MODE}] low-alpha Splash N={splash_low.sum():,} '
      f'({100 * splash_low.sum() / low.sum():.1f}% of low-alpha), '
      f'high-alpha Splash N={splash_high.sum():,} '
      f'({100 * splash_high.sum() / high.sum():.1f}% of high-alpha)')


def track(mask):
    """Median V_phi per snapshot, with the uncertainty *on the median*.

    The band is the 16-84 range of NBOOT bootstrap medians, not the 16-84 range
    of the stars themselves: the question the panel asks is whether the four
    tracks are separated, and for that the error on each median is the relevant
    interval.  The population spread is an order of magnitude wider and, drawn as
    shading, buried the separation it was meant to support.  Bootstrapping rather
    than 1.253*sigma/sqrt(N) because these distributions are skewed, most of all
    the low-alpha Splash, which is a tail of its parent population.
    """
    off = np.concatenate([[0], np.cumsum(counts)])
    med, lo, hi = (np.full(len(times), np.nan) for _ in range(3))
    for k, n in enumerate(counts):
        v = d['snap_vphi'][off[k]:off[k] + n][mask[:n]]
        if len(v) >= NMIN:
            med[k] = np.median(v)
            b = np.median(RNG.choice(v, (NBOOT, len(v))), axis=1)
            lo[k], hi[k] = np.percentile(b, BAND)
            lo[k] = med[k] - ERR_SCALE * (med[k] - lo[k])
            hi[k] = med[k] + ERR_SCALE * (hi[k] - med[k])
    return med, lo, hi


fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(16.8, 4.7))

# --------------------------------------------------- left: selection plane ---
# Only the stars the analysis uses: the [Fe/H] < -1 tail is not plotted.
shown = insitu & (R > RMIN) & (feh > FEH_MIN)
axL.hist2d(feh[shown], ofe[shown], bins=(130, 120),
           range=((FEH_MIN, 0.75), (-0.45, 0.6)), norm=LogNorm(), cmap='Greys',
           rasterized=True)
for y in (OFE_LOW, OFE_HIGH):
    axL.axhline(y, color='#b2182b', lw=1.4, ls='--')
# Hugging the boundaries so each label reads against its own dashed line, but
# pushed to the metal-rich edge where the density map has thinned out.
for y, lab, c, va in [(OFE_HIGH + .04, r'high-$\alpha$', C_HIGH, 'bottom'),
                      (OFE_LOW - .04, r'low-$\alpha$', C_LOW, 'top')]:
    axL.text(0.72, y, lab, color=c, fontsize=19, ha='right', va=va)
# The cuts that shape the sample but are not otherwise visible in the panel.
axL.text(.975, .035, r'$R_{\rm GC}>5$ kpc,  $t_{\rm form}<4$ Gyr,  [Fe/H] $>-1$',
         transform=axL.transAxes, ha='right', fontsize=10.5, color='.35')
axL.set(xlim=(FEH_MIN, 0.75), ylim=(-0.45, 0.6), xlabel='[Fe/H]', ylabel='[O/Fe]')

# --------------------------------------------- middle: the V_phi selection ---
# Unit-area histograms: the two populations differ by 30 per cent in number and
# it is the shape of the slow-rotating tail that the Splash cut acts on.
bins = np.linspace(*VRNG, 90)
for m, c in [(low, C_LOW), (high, C_HIGH)]:
    axM.hist(vphi0[m], bins=bins, density=True, histtype='step', color=c, lw=2.2,
             rasterized=True)
    axM.hist(vphi0[m], bins=bins, density=True, color=c, alpha=.13, lw=0,
             rasterized=True)
# The Splash cuts.  Black when one line applies to both populations (symmetric
# mode); in the paper's asymmetric mode each line is drawn in the colour of the
# population it cuts, since they are then different velocities.
if MODE == 'symmetric':
    cuts = [(-VPHI_LOW, 'k'), (VPHI_LOW, 'k')]
else:
    cuts = [(VPHI_HIGH, C_HIGH), (VPHI_LOW, C_LOW)]
for x, c in cuts:
    axM.axvline(x, color=c, lw=1.4, ls='--')
axM.set(xlim=VRNG, ylim=(0, None), xlabel=r'$V_\phi$ [km s$^{-1}$]',
        ylabel=r'normalised density')
# No legend: the curves are labelled where they run, in their own colours, as in
# panel (a).  A legend box here has nowhere to sit that is clear of both the tall
# green peak and the cut lines on the left.
for x, y, lab, c in [(.88, .74, r'low-$\alpha$', C_LOW),
                     (.50, .585, r'high-$\alpha$', C_HIGH)]:
    axM.text(x, y, lab, transform=axM.transAxes, color=c, fontsize=17,
             ha='center', va='center')

# ------------------------------------------------- right: V_phi against time ---
for m, c, lab, ls in [(low & ~splash_low, C_DISC, r'canonical low-$\alpha$ disc', '-'),
                      (high & ~splash_high, C_DISC, r'canonical high-$\alpha$ disc', '--'),
                      (splash_low, C_LOW, r'low-$\alpha$ Splash', '-'),
                      (splash_high, C_HIGH, r'high-$\alpha$ Splash', '-')]:
    med, lo, hi = track(m)
    axR.fill_between(times, lo, hi, color=c, alpha=.45, lw=0, rasterized=True)
    axR.plot(times, med, color=c, lw=2.4, ls=ls, label=lab, rasterized=True)
    print(f'  {lab:28s} N={m.sum():7,}  median bootstrap 16-84 width '
          f'{np.nanmedian(hi - lo) / ERR_SCALE:5.1f} km/s '
          f'(drawn x{ERR_SCALE})')

# Inflated by ERR_SCALE like the tracks, so the two are read on the same scale.
axR.axhspan(V_EOS - ERR_SCALE * V_EOS_ERR, V_EOS + ERR_SCALE * V_EOS_ERR,
            color=C_EOS, alpha=.22, lw=0, rasterized=True)
axR.axhline(V_EOS, color=C_EOS, lw=1.8, ls='-.')
# Labelled on the line rather than in the legend: as a legend entry it forced the
# box up into the tracks, and the line itself then ran straight through the text.
axR.text(.15, V_EOS + ERR_SCALE * V_EOS_ERR + 8, 'Eos (observed)', color=C_EOS,
         fontsize=11.5,
         ha='left', va='bottom', bbox=dict(fc='white', ec='none', alpha=.8, pad=2))
for t, lab in PERI:
    axR.axvline(t, color='k', lw=1.0)
    # At the foot of each line, not the head: the top of the panel is where the
    # two canonical-disc tracks run, and the labels sat across them.  The bottom
    # left is empty -- the legend is on the right and every track is above zero.
    axR.text(t - .13, .035, lab, transform=axR.get_xaxis_transform(), rotation=90,
             ha='right', va='bottom', fontsize=10.5)
# The lower limit is dropped well below the data so the legend sits entirely
# beneath the Eos line instead of across it.
axR.set(xlim=(0, 10), ylim=(-175, 300), xlabel='Time [Gyr]',
        ylabel=r'$V_\phi$ [km s$^{-1}$]')
axR.legend(loc='lower right', frameon=False, handlelength=2.4, borderaxespad=.8)

for ax, tag in ((axL, '(a)'), (axM, '(b)'), (axR, '(c)')):
    ax.text(.025, .955, tag, transform=ax.transAxes, va='top', fontsize=16,
            fontweight='bold')

fig.tight_layout()
for ext in ('png', 'pdf'):
    out = f'{FIG_DIR}/splash_vphi_evolution_3panel{SUFFIX}.{ext}'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print('saved', out)
