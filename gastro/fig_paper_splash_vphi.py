"""PUBLICATION FIGURE: V_phi evolution of the low- and high-alpha Splash.

Clumpy+merger (GASTRO c.r.c03), after Borbolato et al. (2026) Figure 5 bottom row,
but with a single symmetric Splash window applied to both alpha populations --
|V_phi| < 80 km/s, this project's observational mask (SPLASH_VTAN_MAX in
../src/eos/config.py) -- so that low- and high-alpha are selected like for like.

  left   the [O/Fe]-[Fe/H] plane with the alpha boundaries drawn on it
  right  median V_phi against time for the two Splash populations and their
         canonical discs, with the dwarf's three pericentric passages marked

Writes PNG and PDF at 300 dpi with a tight bounding box into ../Fig_paper/.
Everything that carries data is rasterised; axes, text and lines stay vector.

This is the publication cut of ana_gastro_fig5.py -- keep that one for the
diagnostic version (panel titles, the paper's own asymmetric cuts, the reference
values read off their figure).  Run with `paper` to use their cuts instead.

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
FEH_WINDOW = (-0.7, -0.2)            # window the alpha split is measured in
NMIN, BAND = 15, (16, 84)
PERI = [(1.6, '1st pericentre'), (2.5, '2nd pericentre'), (3.2, 'dwarf disrupted')]
C_LOW, C_HIGH, C_DISC = '#1a9850', '#e08214', '0.15'

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
print(f'[{MODE}] low-alpha Splash N={splash_low.sum():,}, '
      f'high-alpha Splash N={splash_high.sum():,}')


def track(mask):
    off = np.concatenate([[0], np.cumsum(counts)])
    med, lo, hi = (np.full(len(times), np.nan) for _ in range(3))
    for k, n in enumerate(counts):
        v = d['snap_vphi'][off[k]:off[k] + n][mask[:n]]
        if len(v) >= NMIN:
            med[k] = np.median(v)
            lo[k], hi[k] = np.percentile(v, BAND)
    return med, lo, hi


fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.6))

# --------------------------------------------------- left: selection plane ---
# Only the stars the analysis uses: the [Fe/H] < -1 tail is not plotted.
shown = insitu & (R > RMIN) & (feh > FEH_MIN)
axL.hist2d(feh[shown], ofe[shown], bins=(130, 120),
           range=((FEH_MIN, 0.75), (-0.45, 0.6)), norm=LogNorm(), cmap='Greys',
           rasterized=True)
for y in (OFE_LOW, OFE_HIGH):
    axL.axhline(y, color='#b2182b', lw=1.4, ls='--')
for x in FEH_WINDOW:
    axL.axvline(x, color='k', lw=1.0)
# Parked at the metal-rich edge, well inside each region but clear of the
# density map, which thins out beyond [Fe/H] ~ 0.2.
for y, lab, c in [(0.33, r'high-$\alpha$', C_HIGH), (-0.37, r'low-$\alpha$', C_LOW)]:
    axL.text(0.72, y, lab, color=c, fontsize=19, ha='right', va='center')
axL.set(xlim=(FEH_MIN, 0.75), ylim=(-0.45, 0.6), xlabel='[Fe/H]', ylabel='[O/Fe]')

# ------------------------------------------------ right: V_phi against time ---
for m, c, lab, ls in [(low & ~splash_low, C_DISC, r'canonical low-$\alpha$ disc', '-'),
                      (high & ~splash_high, C_DISC, r'canonical high-$\alpha$ disc', '--'),
                      (splash_low, C_LOW, r'low-$\alpha$ Splash', '-'),
                      (splash_high, C_HIGH, r'high-$\alpha$ Splash', '-')]:
    med, lo, hi = track(m)
    axR.fill_between(times, lo, hi, color=c, alpha=.13, lw=0, rasterized=True)
    axR.plot(times, med, color=c, lw=2.4, ls=ls, label=lab, rasterized=True)

for t, lab in PERI:
    axR.axvline(t, color='k', lw=1.0)
    axR.text(t - .13, 292, lab, rotation=90, ha='right', va='top', fontsize=10.5)
axR.set(xlim=(0, 10), ylim=(-50, 300), xlabel='Time [Gyr]',
        ylabel=r'$V_\phi$ [km s$^{-1}$]')
axR.legend(loc='lower right', frameon=False, handlelength=2.4, borderaxespad=.8)

fig.tight_layout()
for ext in ('png', 'pdf'):
    out = f'{FIG_DIR}/splash_vphi_evolution{SUFFIX}.{ext}'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print('saved', out)
