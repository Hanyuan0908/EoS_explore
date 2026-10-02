"""Where did the GS/E progenitor's gas go, and was it always metal-poor?

The gas metallicity map at snapshot 72 shows almost no gas inside the GS/E
stellar contour, and what is there is metal-poor.  Three readings are possible
and this separates them:

  1  the gas was stripped before pericentre         -> gas mass in the satellite
                                                       aperture falls with time
  2  the satellite gas was always metal-poor        -> its [Fe/H] is flat and low
                                                       at every snapshot
  3  the metal-rich gas left and the poor stayed    -> the stripped gas is more
                                                       metal-rich than the retained

Two measurements, over snapshots 62-74 (t = 3.46-5.30 Gyr):

APERTURE TRACK.  At each snapshot the GS/E stellar centroid is the median
position of the clean-debris IDs, and everything within R_AP of it is "in the
satellite".  Reports gas mass, its mass-weighted median [Fe/H], the satellite
stellar mass in the same aperture, and the centroid's distance from the host.

ID TRACK.  The gas cells inside the aperture at the FIRST snapshot are recorded
by ParticleID and followed.  At each later snapshot each tracked ID is one of:
still gas in the aperture / gas outside it / turned into a star / gone (merged
away by derefinement, or a wind particle).  Arepo star particles inherit the ID
of the gas cell they form from, so the star channel is real and not a loss.
Reports the [Fe/H] of the retained gas against that of the escaped gas, which is
the measurement that decides reading 3.

Writes out/gse_gas_stripping.npz and prints both tables.
"""
import gc, os, sys
import numpy as np
import auriga_public as ap
import config_au18 as C

SNAPS = list(range(62, 75))
R_AP = 6.0                      # the satellite aperture, as in the N figures
OUT = C.OUT_DIR + '/gse_gas_stripping.npz'


def wmedian(x, w):
    o = np.argsort(x); x, w = x[o], w[o]
    c = np.cumsum(w)
    return float(np.interp(.5 * c[-1], c, x)) if len(x) else np.nan


GSE = np.sort(np.load(C.OUT_DIR + '/gse_clean_ids.npy'))
st = np.load(C.OUT_DIR + '/snapshot_times.npz')
SN_ALL, T_ALL = st['snaps'], st['t_snap']

rows, tracked = [], None
print(f'{"snap":>5s} {"t":>6s} {"d_host":>7s} {"M*_sat":>10s} {"M_gas":>10s} '
      f'{"f_gas":>6s} {"[Fe/H]gas":>10s} | {"keptGas":>8s} {"escGas":>8s} '
      f'{"->star":>8s} {"lost":>7s} | {"FeH_kept":>9s} {"FeH_esc":>9s}')
for sn in SNAPS:
    t = float(T_ALL[np.flatnonzero(SN_ALL == sn)[0]])
    sub = ap.subhalos.subfind(sn, directory=C.SIM_DIR,
                              loadlist=['SubhaloPos', 'Group_R_Crit200'])
    cen = sub.data['SubhaloPos'][0]

    s = ap.snapshot.load_snapshot(sn, 4, snappath=C.SIM_DIR, verbose=False,
        loadlist=['Coordinates', 'Masses', 'ParticleIDs'])
    s = ap.util.CentreOnHalo(s, cen)
    sp = s.data['Coordinates'] * 1e3
    sid, sm = s.data['ParticleIDs'], s.data['Masses'] * C.MASS_TO_MSUN
    del s; gc.collect()

    o = np.argsort(sid); ss = sid[o]
    p = np.searchsorted(ss, GSE)
    ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == GSE)
    Gp, Gm = sp[o[p[ok]]], sm[o[p[ok]]]
    gcen = np.median(Gp, axis=0)
    d_host = float(np.linalg.norm(gcen))
    in_ap_s = np.linalg.norm(Gp - gcen, axis=1) < R_AP
    Mstar = float(Gm[in_ap_s].sum())

    g = ap.snapshot.load_snapshot(sn, 0, snappath=C.SIM_DIR, verbose=False,
        loadlist=['Coordinates', 'Masses', 'ParticleIDs', 'GFM_Metals'])
    g = ap.util.CentreOnHalo(g, cen)
    gp = g.data['Coordinates'] * 1e3
    gid = g.data['ParticleIDs']
    gm = g.data['Masses'] * C.MASS_TO_MSUN
    feh = C.bracket_abundance(g.data['GFM_Metals'], 'Fe', 'H')
    del g; gc.collect()

    d_g = np.linalg.norm(gp - gcen, axis=1)
    in_ap = d_g < R_AP
    fin = np.isfinite(feh)
    Mgas = float(gm[in_ap].sum())
    feh_ap = wmedian(feh[in_ap & fin], gm[in_ap & fin])

    if tracked is None:
        tracked = np.sort(gid[in_ap])
        kept = esc = star = lost = np.nan
        fk = fe = np.nan
        print(f'  (tracking {len(tracked):,} gas cells from snapshot {sn})')
    else:
        og = np.argsort(gid); gs_ = gid[og]
        q = np.searchsorted(gs_, tracked)
        hit = (q < len(gs_)) & (gs_[np.minimum(q, len(gs_) - 1)] == tracked)
        idx = og[q[hit]]
        still_in = in_ap[idx]
        kept, esc = int(still_in.sum()), int((~still_in).sum())
        f_ok = fin[idx]
        fk = wmedian(feh[idx][still_in & f_ok], gm[idx][still_in & f_ok])
        fe = wmedian(feh[idx][~still_in & f_ok], gm[idx][~still_in & f_ok])
        q2 = np.searchsorted(ss, tracked)
        hit2 = (q2 < len(ss)) & (ss[np.minimum(q2, len(ss) - 1)] == tracked)
        star = int(hit2.sum())
        lost = int(len(tracked) - hit.sum() - star)

    fg = Mgas / (Mgas + Mstar) if (Mgas + Mstar) > 0 else np.nan
    rows.append((sn, t, d_host, Mstar, Mgas, fg, feh_ap, kept, esc, star, lost, fk, fe))
    f = lambda v, w=8, d=0: (f'{v:>{w}.{d}f}' if np.isfinite(v) else f'{"--":>{w}s}')
    print(f'{sn:>5d} {t:>6.2f} {d_host:>7.1f} {Mstar:>10.3e} {Mgas:>10.3e} '
          f'{fg:>6.3f} {feh_ap:>+10.3f} | {f(kept)} {f(esc)} {f(star)} {f(lost,7)} | '
          f'{f(fk,9,3)} {f(fe,9,3)}', flush=True)
    del sp, sid, sm, gp, gid, gm, feh; gc.collect()

R = np.array([[np.nan if v is None else v for v in r] for r in rows], float)
np.savez(OUT, cols=np.array(['snap','t','d_host','Mstar','Mgas','fgas','feh_ap',
                             'kept','esc','star','lost','feh_kept','feh_esc']), data=R)
print(f'\nsaved {OUT}')
