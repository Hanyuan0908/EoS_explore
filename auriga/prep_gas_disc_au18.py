"""Time evolution of the gaseous disc of Auriga halo 18, through the GS/E merger.

The Au18 counterpart of ../gastro/prep_gas_disc.py, so the two simulations can be
read side by side.

Disc-gas definition: **star-forming cells** (StarFormationRate > 0).  A
temperature cut would be the obvious analogue of the gastro measurement but is
wrong here -- Auriga puts star-forming gas on the Springel & Hernquist effective
equation of state, so its InternalEnergy is a pressure floor rather than a real
temperature (only 66% of SF cells at z=0 fall below 3e4 K).  A cold-or-SF variant
is recorded alongside as a check.

Frame: the galaxy is centred on the main subhalo and the rotation is taken from
the stars inside 10 kpc, exactly as util.align_galaxy does, then applied by hand
to the gas so both components share one frame.  align_galaxy puts the disc
angular momentum on component 0, so the disc plane is components (1,2).

Also records, per snapshot, the azimuthally averaged surface-density profile
Sigma(R) of the star-forming gas and of the stars, and the exponential scale
length R_d fitted to it.  R_d is the disc size the observational literature
quotes; R_1/2 is what a simulation measures most robustly.  They are not the same
number -- for a pure exponential R_1/2 = 1.68 R_d -- and they do not have to move
together, since R_1/2 responds to any mass redistribution while R_d is fitted over
a restricted radial range and ignores the centre.

The same two sizes are measured for the **newly formed stars** -- those with
t_snap - t_form < 100 Myr -- which is the stellar counterpart of the star-forming
gas: it says where star formation actually happened over the last 100 Myr rather
than where the gas eligible to form stars sits.  The two need not agree, and
where they do not, the gas is eligible over a wider area than it is actually
using.  Masses are the current particle masses, not GFM_InitialMass; over 100 Myr
mass loss is ~20 per cent and near enough uniform across the population, so it
shifts the normalisation of Sigma(R) and not its shape.

The profiles themselves are saved, not just the fitted numbers, so the fit range
can be changed later without another pass over 78 snapshots.

Writes out/gas_disc_evolution_au18.npz.
"""
import gc, os, sys
import numpy as np
import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod, util

RMAX, ZMAX = 30., 5.
# Sigma(R) bins, and the fit range for R_d in units of that component's own
# R_1/2.  A fixed radial window would be wrong: the Au18 star-forming disc grows
# by a factor of ~4 across the run, so a window that is the disc at t = 2 Gyr is
# the inner third of it at t = 13.  For a pure exponential R_1/2 = 1.68 R_d, so
# [0.5, 2.5] R_1/2 is [0.84, 4.2] R_d -- the usual 1-4 scale-length fit region,
# tracking the disc as it grows.
RBINS = np.arange(0., 30.01, .25)
FIT_LO, FIT_HI = 0.5, 2.5
NEW_GYR = 0.1                        # "newly formed" = the last 100 Myr
# a -> cosmic time by interpolation: the astropy call is an integral, and doing
# it per star for a million stars a snapshot dominates the runtime.
_AGRID = np.linspace(1e-3, 1., 4000)
_TGRID = C.a_to_age(_AGRID)
RMAX_W, ZMAX_W = 50., 10.
T_COLD = 3e4
XH, GAMMA, MP, KB = 0.76, 5. / 3., 1.6726219e-24, 1.380649e-16
SNAPS = [int(x) for x in sys.argv[1:]] or list(range(50, 128))
os.makedirs(C.OUT_DIR, exist_ok=True)


def half_mass(R, m, frac=0.5):
    if len(R) < 20 or m.sum() <= 0:
        return np.nan
    o = np.argsort(R)
    c = np.cumsum(m[o])
    return float(R[o][np.searchsorted(c, frac * c[-1])])


def sigma_profile(R, m, edges=RBINS):
    """Azimuthally averaged surface density [Msun/kpc^2] in fixed radial bins."""
    h = np.histogram(R, bins=edges, weights=m)[0]
    return h / (np.pi * (edges[1:] ** 2 - edges[:-1] ** 2))


def exp_scale(sig, rh, edges=RBINS, lo=FIT_LO, hi=FIT_HI):
    """Exponential scale length from a straight-line fit to ln Sigma(R).

    Fitted over [lo, hi] x R_1/2 and only where Sigma > 0, so empty outer bins
    cannot pull the slope.  Returns NaN rather than a number from fewer than five
    bins or from a rising profile (a positive slope is not a disc).
    """
    if not np.isfinite(rh) or rh <= 0:
        return np.nan
    ctr = .5 * (edges[:-1] + edges[1:])
    ok = (ctr > lo * rh) & (ctr < hi * rh) & (sig > 0)
    if ok.sum() < 5:
        return np.nan
    slope = np.polyfit(ctr[ok], np.log(sig[ok]), 1)[0]
    return float(-1. / slope) if slope < 0 else np.nan


def frame(sn):
    """Centred, disc-aligned stars and gas for one snapshot, in kpc and km/s."""
    sf = sub_mod.subfind(sn, directory=C.SIM_DIR, loadlist=['GroupFirstSub', 'SubhaloPos'])
    cen = sf.data['SubhaloPos'][int(sf.data['GroupFirstSub'][0])]

    st = snap_mod.load_snapshot(sn, 4, snappath=C.SIM_DIR,
        loadlist=['Coordinates', 'Velocities', 'Masses', 'GFM_StellarFormationTime'])
    a = float(st.time)
    real = st.data['GFM_StellarFormationTime'] > 0          # drop wind particles
    for k in list(st.data):
        st.data[k] = st.data[k][real]
    util.CentreOnHalo(st, cen)
    r = np.sqrt((st.data['Coordinates'] ** 2).sum(1))
    inner = r < .01
    if inner.sum() < 100:
        return None
    bulk = np.average(st.data['Velocities'][inner], axis=0, weights=st.data['Masses'][inner])
    st.data['Velocities'] -= bulk

    # Reproduce align_galaxy's rotation, but keep the axes so the gas can share it.
    idx = np.flatnonzero(r < .01)
    L = np.cross(st.data['Coordinates'][idx, :],
                 st.data['Velocities'][idx, :] * st.data['Masses'][idx, None]).sum(axis=0)
    Ldir = L / np.sqrt((L ** 2).sum())
    xdir, ydir, zdir = util.get_principal_axis(st, idx, L=Ldir)
    util.rotateto(st, xdir, dir2=ydir, dir3=zdir)

    gs = snap_mod.load_snapshot(sn, 0, snappath=C.SIM_DIR,
        loadlist=['Coordinates', 'Velocities', 'Masses', 'StarFormationRate',
                  'InternalEnergy', 'ElectronAbundance'])
    util.CentreOnHalo(gs, cen)
    gs.data['Velocities'] -= bulk
    util.rotateto(gs, xdir, dir2=ydir, dir3=zdir)
    return st, gs, a


rec = {k: [] for k in ('snap', 'a', 'time', 'rhalf_sf', 'rhalf_coldsf', 'rhalf_sf_wide',
                       'rhalf_star', 'r90_sf', 'm_sf', 'm_coldsf', 'm_star', 'sfr',
                       'm_sf_outside', 'zabs_sf', 'rd_sf', 'rd_star',
                       'rhalf_new', 'rd_new', 'm_new', 'n_new')}
prof = {'sigma_sf': [], 'sigma_star': [], 'sigma_new': []}
for sn in SNAPS:
    try:
        got = frame(sn)
    except Exception as exc:
        print(f'  snap {sn}: SKIP ({type(exc).__name__}: {exc})', flush=True)
        continue
    if got is None:
        print(f'  snap {sn}: SKIP (too few central stars)', flush=True)
        continue
    st, gs, a = got

    gx = gs.data['Coordinates'] * 1000.
    Rg = np.hypot(gx[:, 1], gx[:, 2])
    zg = gx[:, 0]
    gm = gs.data['Masses'] * C.MASS_TO_MSUN
    sfr = np.asarray(gs.data['StarFormationRate'], float)
    xe = np.asarray(gs.data['ElectronAbundance'], float)
    u = np.asarray(gs.data['InternalEnergy'], float)
    T = (GAMMA - 1) * u * 1e10 * (4.0 / (1 + 3 * XH + 4 * XH * xe) * MP) / KB

    sx = st.data['Coordinates'] * 1000.
    Rs = np.hypot(sx[:, 1], sx[:, 2])
    zs = sx[:, 0]
    sm = st.data['Masses'] * C.MASS_TO_MSUN

    tform = np.interp(st.data['GFM_StellarFormationTime'], _AGRID, _TGRID)
    tnow = float(C.a_to_age(a))

    ap = (Rg < RMAX) & (np.abs(zg) < ZMAX)
    apw = (Rg < RMAX_W) & (np.abs(zg) < ZMAX_W)
    sfg = ap & (sfr > 0)
    coldsf = ap & ((sfr > 0) | (T < T_COLD))
    star = (Rs < RMAX) & (np.abs(zs) < ZMAX)
    new = star & (tnow - tform < NEW_GYR) & (tform <= tnow)

    rec['snap'].append(sn); rec['a'].append(a); rec['time'].append(float(C.a_to_age(a)))
    rec['rhalf_sf'].append(half_mass(Rg[sfg], gm[sfg]))
    rec['rhalf_coldsf'].append(half_mass(Rg[coldsf], gm[coldsf]))
    rec['rhalf_sf_wide'].append(half_mass(Rg[apw & (sfr > 0)], gm[apw & (sfr > 0)]))
    rec['rhalf_star'].append(half_mass(Rs[star], sm[star]))
    rec['r90_sf'].append(half_mass(Rg[sfg], gm[sfg], .9))
    rec['m_sf'].append(gm[sfg].sum()); rec['m_coldsf'].append(gm[coldsf].sum())
    rec['m_star'].append(sm[star].sum()); rec['sfr'].append(sfr[sfg].sum())
    rec['m_sf_outside'].append(gm[(sfr > 0) & ~ap].sum())
    rec['zabs_sf'].append(float(np.median(np.abs(zg[sfg]))) if sfg.sum() > 20 else np.nan)

    rec['rhalf_new'].append(half_mass(Rs[new], sm[new]))
    rec['m_new'].append(sm[new].sum()); rec['n_new'].append(int(new.sum()))

    ssf = sigma_profile(Rg[sfg], gm[sfg])
    sst = sigma_profile(Rs[star], sm[star])
    snw = sigma_profile(Rs[new], sm[new])
    prof['sigma_sf'].append(ssf); prof['sigma_star'].append(sst)
    prof['sigma_new'].append(snw)
    rec['rd_sf'].append(exp_scale(ssf, rec['rhalf_sf'][-1]))
    rec['rd_star'].append(exp_scale(sst, rec['rhalf_star'][-1]))
    rec['rd_new'].append(exp_scale(snw, rec['rhalf_new'][-1]))
    print(f"  snap {sn:3d}  a={a:.4f}  t={rec['time'][-1]:5.2f} Gyr  "
          f"R_half(SF)={rec['rhalf_sf'][-1]:6.2f} kpc  R_d(SF)={rec['rd_sf'][-1]:6.2f} kpc  "
          f"new*({rec['n_new'][-1]:6,d}) R_1/2={rec['rhalf_new'][-1]:5.2f} "
          f"R_d={rec['rd_new'][-1]:5.2f}  "
          f"M_SF={rec['m_sf'][-1]:.2e}  SFR={rec['sfr'][-1]:6.2f}", flush=True)
    del st, gs, gx, sx
    gc.collect()

np.savez(C.OUT_DIR + '/gas_disc_evolution_au18.npz',
         rbins=RBINS, fit_range=np.array([FIT_LO, FIT_HI]), new_gyr=NEW_GYR,
         **{k: np.array(v) for k, v in prof.items()},
         **{k: np.array(v) for k, v in rec.items()})
print('\nsaved', C.OUT_DIR + '/gas_disc_evolution_au18.npz')
