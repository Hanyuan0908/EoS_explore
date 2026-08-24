"""Time evolution of the gaseous disc of the Clumpy+merger model.

Question: does the size of the gas disc respond to the GSE-like merger?

The gas is 96% hot halo by mass, so "the disc" has to be a cold-gas selection:
T < 3e4 K inside R < 30 kpc and |z| < 3 kpc.  At z=0 that picks 3.2e9 Msol in a
very thin layer (90th percentile |z| = 0.25 kpc), and the resulting half-mass
radius is insensitive to the aperture (4.12 kpc for R<30/|z|<3 and R<50/|z|<5
alike), so the measurement is not an artefact of where the box edges are put.

Recorded per snapshot, several variants so the trend can be checked rather than
taken on faith:
  rhalf_cold   T < 3e4 K   -- the headline definition
  rhalf_cool   T < 1e4 K   -- colder, closer to star-forming gas
  rhalf_corot  T < 3e4 K and v_phi > 50 km/s -- rejects the satellite's own gas,
               which is on an inclined orbit and does not co-rotate with the disc
  rhalf_star   stars, same aperture -- for comparison
plus the enclosed masses, r90, and the cold gas beyond the aperture (a tracer of
the satellite arriving).

Writes out/gas_disc_evolution.npz.
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
T_COLD, T_COOL = 3e4, 1e4
# Star-formation criteria.  The run's own .param is not available, so GASOLINE's
# thresholds cannot be read off directly; these are the two standard choices and
# they bracket it.  AGERTZ is the VINTERGATAN definition (Agertz et al. 2021,
# their Fig. 6): T < 1e4 K and n_H > 1 cm^-3.  GASOLINE_STD is the code's usual
# pair, T < 1.5e4 K and n_H > 0.1 cm^-3.  The gas here reaches n_H ~ 80 cm^-3, so
# both are well resolved -- unlike Auriga, where star-forming gas sits at
# n_H ~ 0.1-0.5 and an n > 1 cut would select only the nucleus.
T_AGERTZ, N_AGERTZ = 1e4, 1.0
T_GASOLINE, N_GASOLINE = 1.5e4, 0.1
MSUN, KPC, XH, MP = 1.989e33, 3.0857e21, 0.76, 1.6726219e-24
RMAX, ZMAX = 30., 3.
VPHI_COROT = 50.
os.makedirs(G.OUT_DIR, exist_ok=True)


def half_mass(R, m, frac=0.5):
    if len(R) < 20 or m.sum() <= 0:
        return np.nan
    o = np.argsort(R)
    c = np.cumsum(m[o])
    return float(R[o][np.searchsorted(c, frac * c[-1])])


files = sorted(glob.glob(MODEL_DIR + f'/jrun003.{NAME}.0*'))
files = [x for x in files if not any(k in x for k in ('MassFrac', 'iord', 'timeform'))]

rec = {k: [] for k in ('time', 'rhalf_cold', 'rhalf_cool', 'rhalf_corot', 'rhalf_star',
                       'r90_cold', 'm_cold', 'm_cool', 'm_star', 'm_cold_outside',
                       'vphi_cold', 'zabs_cold',
                       'rhalf_agertz', 'm_agertz', 'r90_agertz',
                       'rhalf_gasoline', 'm_gasoline')}
for path in files:
    f = pynbody.load(path)
    f.physical_units()
    pynbody.analysis.angmom.faceon(f.stars)

    gp = np.asarray(f.g['pos'], float)
    gm = np.asarray(f.g['mass'], float)
    T = np.asarray(f.g['temp'], float)
    nH = np.asarray(f.g['rho'], float) * MSUN / KPC ** 3 * XH / MP
    gv = np.asarray(f.g['vel'], float)
    Rg = np.hypot(gp[:, 0], gp[:, 1])
    zg = gp[:, 2]
    safe = np.where(Rg > .1, Rg, 1.)
    vphi = (gp[:, 0] * gv[:, 1] - gp[:, 1] * gv[:, 0]) / safe

    sp = np.asarray(f.s['pos'], float)
    sm = np.asarray(f.s['mass'], float)
    Rs = np.hypot(sp[:, 0], sp[:, 1])
    zs = sp[:, 2]
    # The stellar disc sets the sense of rotation; flip gas to match if needed.
    disc = (Rs > 2) & (Rs < 8) & (np.abs(zs) < 2)
    sv = np.asarray(f.s['vel'], float)
    ssafe = np.where(Rs > .1, Rs, 1.)
    vphi_s = (sp[:, 0] * sv[:, 1] - sp[:, 1] * sv[:, 0]) / ssafe
    if disc.sum() > 100 and np.median(vphi_s[disc]) < 0:
        vphi = -vphi

    ap = (Rg < RMAX) & (np.abs(zg) < ZMAX)
    cold = ap & (T < T_COLD)
    cool = ap & (T < T_COOL)
    corot = cold & (vphi > VPHI_COROT)
    agertz = ap & (T < T_AGERTZ) & (nH > N_AGERTZ)
    gasoline = ap & (T < T_GASOLINE) & (nH > N_GASOLINE)
    star = (Rs < RMAX) & (np.abs(zs) < ZMAX)

    rec['time'].append(float(f.properties['time']))
    rec['rhalf_cold'].append(half_mass(Rg[cold], gm[cold]))
    rec['rhalf_cool'].append(half_mass(Rg[cool], gm[cool]))
    rec['rhalf_corot'].append(half_mass(Rg[corot], gm[corot]))
    rec['rhalf_star'].append(half_mass(Rs[star], sm[star]))
    rec['r90_cold'].append(half_mass(Rg[cold], gm[cold], .9))
    rec['m_cold'].append(gm[cold].sum())
    rec['m_cool'].append(gm[cool].sum())
    rec['m_star'].append(sm[star].sum())
    rec['m_cold_outside'].append(gm[(T < T_COLD) & ~ap].sum())
    rec['vphi_cold'].append(float(np.median(vphi[cold])) if cold.sum() > 20 else np.nan)
    rec['rhalf_agertz'].append(half_mass(Rg[agertz], gm[agertz]))
    rec['r90_agertz'].append(half_mass(Rg[agertz], gm[agertz], .9))
    rec['m_agertz'].append(gm[agertz].sum())
    rec['rhalf_gasoline'].append(half_mass(Rg[gasoline], gm[gasoline]))
    rec['m_gasoline'].append(gm[gasoline].sum())
    rec['zabs_cold'].append(float(np.median(np.abs(zg[cold]))) if cold.sum() > 20 else np.nan)
    print(f"  t={rec['time'][-1]:5.2f}  R_half(cold)={rec['rhalf_cold'][-1]:6.2f} kpc  "
          f"R_half(SF,Agertz)={rec['rhalf_agertz'][-1]:6.2f}  "
          f"M_SF={rec['m_agertz'][-1]:.2e}", flush=True)
    del f

# Star formation history, straight from the birth times in the final snapshot.
f = pynbody.load(MODEL_DIR + f'/jrun003.{NAME}.01000')
f.physical_units()
tf = np.asarray(f.s['tform'], float)
mi = np.asarray(f.s['mass'], float)
edges = np.arange(0, 10.05, .1)
sfr = np.histogram(tf, bins=edges, weights=mi)[0] / (.1 * 1e9)   # Msol/yr

np.savez(G.OUT_DIR + '/gas_disc_evolution.npz',
         sfr_edges=edges, sfr=sfr, **{k: np.array(v) for k, v in rec.items()})
print('\nsaved', G.OUT_DIR + '/gas_disc_evolution.npz')
