"""Guiding-centre radius at birth for the merger-born sample.

R_birth is the instantaneous radius at the nearest snapshot, so for an eccentric
orbit it depends on where in the orbit the star happens to be caught.  R_g is the
radius of the circular orbit with the same angular momentum, which does not: it
is the radius the star "belongs to".

The circular velocity comes from the ENCLOSED MASS of every species,
v_c(r) = sqrt(G M(<r) / r).  Two potential-gradient estimates were tried first
and both failed the basic check that disc stars cannot rotate faster than v_c: a
spherically averaged Phi(r) gave v_c(8 kpc) = 162 km/s against a disc rotating at
195, and binning Phi in the plane gave the same number with a gradient too noisy
to invert.  Enclosed mass is unambiguous.  It is still spherical, so it
understates the in-plane force of a flattened disc by roughly ten per cent and
R_g is a slight overestimate -- but uniformly so for both populations, which is
what matters for comparing them.  A circular orbit at R carries
L_circ(R) = R v_c(R), which rises monotonically, so inverting it maps each star's
L_z onto its guiding radius.  Stars on retrograde orbits are mapped through
|L_z| and flagged; R_g is undefined in sign for them.

Writes out/merger_rg_birth.npz.
"""
import gc, os, sys
import numpy as np
import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod, util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import orbit_tools as OT

SNAPS = list(range(73, 83))
os.makedirs(C.OUT_DIR, exist_ok=True)
RGRID = np.logspace(np.log10(0.2), np.log10(120.), 260)


def frame(sn):
    s = snap_mod.load_snapshot(sn, 4, snappath=C.SIM_DIR,
        loadlist=['ParticleIDs', 'Coordinates', 'Velocities', 'Masses', 'Potential',
                  'GFM_StellarFormationTime'])
    real = s.data['GFM_StellarFormationTime'] > 0
    for k in list(s.data):
        s.data[k] = s.data[k][real]
    sf = sub_mod.subfind(sn, directory=C.SIM_DIR, loadlist=['GroupFirstSub', 'SubhaloPos'])
    cen = sf.data['SubhaloPos'][int(sf.data['GroupFirstSub'][0])]
    util.CentreOnHalo(s, cen)
    rr = np.sqrt((s.data['Coordinates'] ** 2).sum(1))
    q = rr < .01
    s.data['Velocities'] -= np.average(s.data['Velocities'][q], axis=0,
                                       weights=s.data['Masses'][q])
    util.align_galaxy(s, radialcut=.01)
    return s, cen


ids, rg, lz, Rb, vphib, snb, retro = [], [], [], [], [], [], []
for sn in SNAPS:
    s, cen = frame(sn)
    a = float(s.time)
    a_prev = float(snap_mod.load_snapshot(sn - 1, 4, snappath=C.SIM_DIR,
                                          loadlist=['GFM_StellarFormationTime']).time)
    x = s.data['Coordinates'] * 1000.
    v = s.data['Velocities']
    r = np.sqrt((x * x).sum(1))
    R = np.hypot(x[:, 1], x[:, 2])
    safe = np.where(R > .1, R, 1.)
    vphi = (x[:, 1] * v[:, 2] - x[:, 2] * v[:, 1]) / safe
    disc = (R > 3) & (R < 12) & (np.abs(x[:, 0]) < 2)
    sign = -1. if np.median(vphi[disc]) < 0 else 1.
    vphi *= sign

    # circular velocity from the enclosed mass of every species
    Menc = np.zeros(len(RGRID))
    for ptype in (0, 1, 2, 3, 4):
        try:
            pp = snap_mod.load_snapshot(sn, ptype, snappath=C.SIM_DIR,
                                        loadlist=['Coordinates', 'Masses'])
        except Exception:
            continue
        if 'Masses' not in pp.data or not len(np.atleast_1d(pp.data['Masses'])):
            continue
        rr_p = np.sqrt(((pp.data['Coordinates'] - cen) ** 2).sum(1)) * 1000.
        mm_p = np.asarray(pp.data['Masses'], float) * C.MASS_TO_MSUN
        cnt_p = np.bincount(np.clip(np.searchsorted(RGRID, rr_p), 0, len(RGRID) - 1),
                            weights=mm_p, minlength=len(RGRID))
        Menc += np.cumsum(cnt_p)
        del pp, rr_p, mm_p
        gc.collect()
    GRAV = 4.30091e-6                                  # kpc (km/s)^2 / Msun
    vc = np.sqrt(GRAV * Menc / RGRID)
    Lcirc = RGRID * vc
    keep = np.concatenate([[True], np.diff(Lcirc) > 0])
    Rg_grid, Lcirc, vc = RGRID[keep], Lcirc[keep], vc[keep]

    born = (s.data['GFM_StellarFormationTime'] > a_prev) & (s.data['GFM_StellarFormationTime'] <= a)
    Lz = R[born] * vphi[born]
    Rg = np.interp(np.abs(Lz), Lcirc, Rg_grid, left=Rg_grid[0], right=np.nan)

    ids.append(s.data['ParticleIDs'][born]); rg.append(Rg); lz.append(Lz)
    Rb.append(R[born]); vphib.append(vphi[born])
    snb.append(np.full(born.sum(), sn)); retro.append(Lz < 0)
    print(f'  snap {sn}: {born.sum():6,} born  v_c(8kpc)={np.interp(8., Rg_grid, vc):6.1f}  '
          f'v_phi(disc)={np.median(vphi[disc]):6.1f}  '
          f'median R_g={np.nanmedian(Rg):5.2f} (R={np.median(R[born]):5.2f})', flush=True)
    del s, x, v; gc.collect()

np.savez(C.OUT_DIR + '/merger_rg_birth.npz',
         ids=np.concatenate(ids), Rg_birth=np.concatenate(rg), Lz_birth=np.concatenate(lz),
         R_birth=np.concatenate(Rb), vphi_birth=np.concatenate(vphib),
         snap_birth=np.concatenate(snb), retrograde=np.concatenate(retro))
print('\nsaved', C.OUT_DIR + '/merger_rg_birth.npz')
