"""Is the gas that leaves the GS/E progenitor more metal-rich than the gas that stays?

diag_gse_gas_stripping tracks the gas mass and metallicity inside the satellite
aperture.  It cannot say whether stripping is metallicity-selective, and its
ParticleID track turned out to be useless for that: gas cell IDs are not
conserved under Arepo's refinement/derefinement, and 60-95 per cent of the
tracked cells vanish within one snapshot.  Those columns are not to be believed.

This measures the same thing without IDs, by comparing gas at the same moment in
three places around the satellite:

  core      |r - r_GSE| <  3 kpc           what the satellite still holds
  envelope  3 <= |r - r_GSE| < 6 kpc       its outskirts, stripped first
  wake      6 <= |r - r_GSE| < 15 kpc, and |r| > 20 kpc from the host centre
                                           material recently removed, before it
                                           has mixed into the host's halo

The host-distance condition on the wake matters: without it the shell sweeps up
host halo gas once the satellite is inside ~25 kpc, and that is a different
reservoir with its own metallicity.  Where the satellite is too close for the
condition to leave anything, the wake column is blank rather than wrong.

All metallicities are MASS-WEIGHTED medians, so a few tiny enriched cells cannot
carry the number.

Writes out/gse_gas_shells.npz.
"""
import gc, os, sys
import numpy as np
import auriga_public as ap
import config_au18 as C

SNAPS = [66, 68, 70, 71, 72, 73, 74]
R_CORE, R_AP, R_WAKE, D_HOST_MIN = 3.0, 6.0, 15.0, 20.0
OUT = C.OUT_DIR + '/gse_gas_shells.npz'


def wmedian(x, w):
    if not len(x):
        return np.nan
    o = np.argsort(x); x, w = x[o], w[o]
    c = np.cumsum(w)
    return float(np.interp(.5 * c[-1], c, x))


GSE = np.sort(np.load(C.OUT_DIR + '/gse_clean_ids.npy'))
st = np.load(C.OUT_DIR + '/snapshot_times.npz')
SN_ALL, T_ALL = st['snaps'], st['t_snap']

print(f'{"snap":>5s} {"t":>6s} {"d_host":>7s} | {"M_core":>9s} {"FeH_core":>9s} | '
      f'{"M_env":>9s} {"FeH_env":>9s} | {"M_wake":>9s} {"FeH_wake":>9s} | '
      f'{"FeH_disc":>9s}')
rows = []
for sn in SNAPS:
    t = float(T_ALL[np.flatnonzero(SN_ALL == sn)[0]])
    sub = ap.subhalos.subfind(sn, directory=C.SIM_DIR, loadlist=['SubhaloPos'])
    cen = sub.data['SubhaloPos'][0]
    s = ap.snapshot.load_snapshot(sn, 4, snappath=C.SIM_DIR, verbose=False,
        loadlist=['Coordinates', 'ParticleIDs'])
    s = ap.util.CentreOnHalo(s, cen)
    sp = s.data['Coordinates'] * 1e3; sid = s.data['ParticleIDs']
    del s; gc.collect()
    o = np.argsort(sid); ss = sid[o]; p = np.searchsorted(ss, GSE)
    ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == GSE)
    gcen = np.median(sp[o[p[ok]]], axis=0)
    d_host = float(np.linalg.norm(gcen))
    del sp, sid; gc.collect()

    g = ap.snapshot.load_snapshot(sn, 0, snappath=C.SIM_DIR, verbose=False,
        loadlist=['Coordinates', 'Masses', 'GFM_Metals'])
    g = ap.util.CentreOnHalo(g, cen)
    gp = g.data['Coordinates'] * 1e3
    gm = g.data['Masses'] * C.MASS_TO_MSUN
    feh = C.bracket_abundance(g.data['GFM_Metals'], 'Fe', 'H')
    del g; gc.collect()

    dg = np.linalg.norm(gp - gcen, axis=1)
    dh = np.linalg.norm(gp, axis=1)
    fin = np.isfinite(feh)
    core = (dg < R_CORE) & fin
    env = (dg >= R_CORE) & (dg < R_AP) & fin
    wake = (dg >= R_AP) & (dg < R_WAKE) & (dh > D_HOST_MIN) & fin
    disc = (np.abs(gp[:, 2]) < 2.) & (np.hypot(gp[:, 0], gp[:, 1]) < 8.) & fin
    out = [sn, t, d_host]
    for m in (core, env, wake):
        out += [float(gm[m].sum()), wmedian(feh[m], gm[m])]
    out.append(wmedian(feh[disc], gm[disc]))
    rows.append(out)
    f = lambda v, d=3: (f'{v:>+9.{d}f}' if np.isfinite(v) else f'{"--":>9s}')
    print(f'{sn:>5d} {t:>6.2f} {d_host:>7.1f} | {out[3]:>9.3e} {f(out[4])} | '
          f'{out[5]:>9.3e} {f(out[6])} | {out[7]:>9.3e} {f(out[8])} | {f(out[9])}',
          flush=True)
    del gp, gm, feh; gc.collect()

np.savez(OUT, cols=np.array(['snap','t','d_host','M_core','feh_core','M_env',
                             'feh_env','M_wake','feh_wake','feh_disc']),
         data=np.array(rows, float))
print(f'\nsaved {OUT}')
