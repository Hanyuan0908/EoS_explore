"""Prep for the Borbolato et al. (2026) Figure 5 reproduction, Clumpy+merger model.

Selects the low- and high-alpha Splash at z=0 following their Section 3.2, then
walks the snapshot series backwards to record <V_phi>(t) for each population.

Their recipe, as implemented here:
  * alpha tracer is **oxygen**: everything in [O/Fe]-[Fe/H];
  * the split is empirical - take -0.7 < [Fe/H] < -0.2, find the dip in the
    [O/Fe] histogram, and (their words) "introduce a gap between the two
    populations to avoid regions where they may overlap" for the clumpy models;
  * radial selection normalised by the disc scale length, matching the
    observational R_GC > 5 kpc with the MW's R_d ~ 2.6 kpc;
  * Splash = halo-like azimuthal velocity: **V_phi < 100 km/s** for low-alpha,
    the stricter **V_phi < 50 km/s** for high-alpha (the simulated high-alpha
    disc is more heated than the MW's, so the loose cut drags in the disc).
  No eccentricity cut is applied - they verify a posteriori that it selects e>0.6.

Cross-snapshot identity uses the fact that GASOLINE appends new stars to the end
of the file, so a star's array index is a stable id; this is verified in the run.

Writes out/fig5_clumpy_merger.npz.
"""
import glob, os, sys
import numpy as np
import pynbody
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gastro_config as G

MODEL_DIR = G.HERE + '/jrun003.dwarfM06XY138Z37Vxy20FB20'
NAME = 'dwarfM06XY138Z37Vxy20FB20'
FEH_WINDOW = (-0.7, -0.2)        # the metallicity slice the alpha split is made in
MW_RD = 2.6                      # kpc, adopted MW disc scale length
MW_RMIN = 5.0                    # kpc, the observational R_GC cut
VPHI_LOW, VPHI_HIGH = 100., 50.  # Splash cuts for low- and high-alpha
RFORM_SAT = 30.                  # kpc: stars born beyond this are the satellite's
os.makedirs(G.OUT_DIR, exist_ok=True)


def aligned(path):
    f = pynbody.load(path)
    f.physical_units()
    pynbody.analysis.angmom.faceon(f.stars)
    return f


def cylindrical(f):
    pos = np.asarray(f.s['pos'], float)
    vel = np.asarray(f.s['vel'], float)
    R = np.hypot(pos[:, 0], pos[:, 1])
    safe = np.where(R > .1, R, 1.)
    vphi = (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]) / safe
    disc = (R > 2) & (R < 8) & (np.abs(pos[:, 2]) < 2)
    if disc.sum() > 100 and np.median(vphi[disc]) < 0:
        vphi = -vphi
    return R, pos[:, 2], vphi


def scale_length(R, z, mass, lo=3., hi=12.):
    """Exponential scale length of the z=0 stellar disc."""
    sel = np.abs(z) < 3
    b = np.arange(0, 20.25, .25)
    area = np.pi * (b[1:] ** 2 - b[:-1] ** 2)
    ib = np.clip(np.searchsorted(b, R[sel]) - 1, 0, len(b) - 2)
    sig = np.bincount(ib, weights=mass[sel], minlength=len(b) - 1) / area
    rc = .5 * (b[:-1] + b[1:])
    fit = (rc > lo) & (rc < hi) & (sig > 0)
    return -1. / np.polyfit(rc[fit], np.log(sig[fit]), 1)[0]


def alpha_split(ofe, smooth=3):
    """The deepest valley in the [O/Fe] histogram: the high/low-alpha boundary.

    Scored by valley *depth* -- min(tallest peak to the left, tallest peak to the
    right) minus the bin itself -- rather than by picking peaks first.  Choosing
    peaks first is fragile here: the tallest maximum below [O/Fe]=0 is the inner
    edge of the high-alpha cloud, not the low-alpha sequence, which sends a
    peak-first detector to a shallow shoulder at +0.05 instead of the real
    bimodal minimum near -0.15.
    """
    bins = np.arange(-0.45, 0.65, 0.01)
    ctr = .5 * (bins[:-1] + bins[1:])
    h, _ = np.histogram(ofe, bins=bins)
    hs = np.convolve(h.astype(float), np.ones(smooth) / smooth, mode='same')
    depth = np.full(len(hs), -np.inf)
    for i in range(2, len(hs) - 2):
        depth[i] = min(hs[:i].max(), hs[i + 1:].max()) - hs[i]
    i = int(np.argmax(depth))
    dip = ctr[i]
    lo_peak = ctr[:i][np.argmax(hs[:i])]
    hi_peak = ctr[i + 1:][np.argmax(hs[i + 1:])]
    print(f'  valley depth at the chosen dip: {depth[i]:.0f} counts '
          f'(peaks {hs[:i].max():.0f} at {lo_peak:+.3f}, {hs[i+1:].max():.0f} at {hi_peak:+.3f})')
    return dip, lo_peak, hi_peak, ctr, hs


# ------------------------------------------------------------------ z = 0 ----
print('loading the z=0 snapshot ...', flush=True)
f0 = aligned(MODEL_DIR + f'/jrun003.{NAME}.01000')
R0, z0, vphi0 = cylindrical(f0)
mass = np.asarray(f0.s['mass'], float)
feh = np.asarray(f0.s['feh'], float)
ofe = np.asarray(f0.s['ofe'], float)
tform = np.asarray(f0.s['tform'], float)
nstar = len(R0)

xf = np.load(f'{MODEL_DIR}/{NAME}_xform.npy')
yf = np.load(f'{MODEL_DIR}/{NAME}_yform.npy')
zf = np.load(f'{MODEL_DIR}/{NAME}_zform.npy')
Rform = np.hypot(xf, yf)

Rd = scale_length(R0, z0, mass)
RMIN = MW_RMIN * Rd / MW_RD
print(f'disc scale length R_d = {Rd:.2f} kpc  ->  R_GC>{MW_RMIN:.0f} kpc maps to R > {RMIN:.2f} kpc')

insitu = Rform < RFORM_SAT
print(f'satellite-born (R_form>{RFORM_SAT:.0f} kpc): {(~insitu).sum():,} stars, '
      f'M*={mass[~insitu].sum():.3e} Msol (paper: 8.97e8 at first pericentre)')

vol = (R0 > RMIN) & insitu
win = vol & (feh > FEH_WINDOW[0]) & (feh < FEH_WINDOW[1])
dip, lo_peak, hi_peak, hctr, hsm = alpha_split(ofe[win])
GAP = 0.02
print(f'[O/Fe] low-alpha peak {lo_peak:+.3f}, high-alpha peak {hi_peak:+.3f}, dip {dip:+.3f}'
      f'  -> split at {dip:+.3f} with a +/-{GAP} gap')

low = vol & (ofe < dip - GAP)
high = vol & (ofe > dip + GAP)
splash_low = low & (vphi0 < VPHI_LOW)
splash_high = high & (vphi0 < VPHI_HIGH)
print(f'low-alpha  {low.sum():,}  -> Splash {splash_low.sum():,} ({100*splash_low.sum()/low.sum():.2f}%)')
print(f'high-alpha {high.sum():,}  -> Splash {splash_high.sum():,} ({100*splash_high.sum()/high.sum():.2f}%)')

# --------------------------------------------------- per-snapshot kinematics ---
# Cache v_phi and R for every star at every snapshot, concatenated with offsets.
# The population masks stay a plotting-time decision, so re-deciding the alpha
# split (or the Splash cuts) never costs another pass over the series.
files = sorted(glob.glob(MODEL_DIR + f'/jrun003.{NAME}.0*'))
files = [x for x in files if not any(k in x for k in ('MassFrac', 'iord', 'timeform'))]

times, counts, vphi_all, R_all = [], [], [], []
for path in files:
    f = aligned(path)
    R, z, vphi = cylindrical(f)
    times.append(float(f.properties['time']))
    counts.append(len(R))
    vphi_all.append(vphi.astype(np.float32))
    R_all.append(R.astype(np.float32))
    med = np.median(vphi[splash_low[:len(R)]]) if splash_low[:len(R)].sum() > 50 else np.nan
    print(f'  {os.path.basename(path):52s} t={times[-1]:5.2f}  N*={len(R):8,}  '
          f'median vphi of the low-alpha Splash = {med:6.1f}', flush=True)
    del f

np.savez(G.OUT_DIR + '/fig5_clumpy_merger.npz',
         tform=tform, Rform=Rform, feh=feh, ofe=ofe, R=R0, z=z0, vphi=vphi0, mass=mass,
         insitu=insitu, low=low, high=high, splash_low=splash_low, splash_high=splash_high,
         times=np.array(times), counts=np.array(counts), Rd=Rd, RMIN=RMIN,
         dip=dip, gap=GAP, ofe_hist_x=hctr, ofe_hist_y=hsm,
         snap_vphi=np.concatenate(vphi_all), snap_R=np.concatenate(R_all))
print('\nsaved', G.OUT_DIR + '/fig5_clumpy_merger.npz')
