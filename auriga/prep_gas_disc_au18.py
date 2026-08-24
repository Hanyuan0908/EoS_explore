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

Writes out/gas_disc_evolution_au18.npz.
"""
import gc, os, sys
import numpy as np
import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod, util

RMAX, ZMAX = 30., 5.
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
                       'm_sf_outside', 'zabs_sf')}
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

    ap = (Rg < RMAX) & (np.abs(zg) < ZMAX)
    apw = (Rg < RMAX_W) & (np.abs(zg) < ZMAX_W)
    sfg = ap & (sfr > 0)
    coldsf = ap & ((sfr > 0) | (T < T_COLD))
    star = (Rs < RMAX) & (np.abs(zs) < ZMAX)

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
    print(f"  snap {sn:3d}  a={a:.4f}  t={rec['time'][-1]:5.2f} Gyr  "
          f"R_half(SF)={rec['rhalf_sf'][-1]:6.2f} kpc  M_SF={rec['m_sf'][-1]:.2e}  "
          f"SFR={rec['sfr'][-1]:6.2f}", flush=True)
    del st, gs, gx, sx
    gc.collect()

np.savez(C.OUT_DIR + '/gas_disc_evolution_au18.npz', **{k: np.array(v) for k, v in rec.items()})
print('\nsaved', C.OUT_DIR + '/gas_disc_evolution_au18.npz')
