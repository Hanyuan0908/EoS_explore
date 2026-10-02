"""Of the mass born on halo orbits during the merger, how much is born in the bridge?

The burst window 4.9-5.5 Gyr spans five stored snapshots, and the bridge is not a
fixed region: it is defined relative to the GS/E stellar centroid, which moves
19.7 -> 34.3 -> 13.6 kpc as the satellite swings through pericentre and dissolves.
So the aperture classification has to be redone AT EVERY SNAPSHOT and the masses
summed, not evaluated once at snapshot 72.

Each star is assigned to the first stored snapshot at or after it formed -- the
same convention au18_birth_classes uses to place stars on a map -- and classified
there:

    GS/E analogue   |r - r_GSE| <  6 kpc        (r_GSE = clean-debris centroid)
    MW analogue     |r - r_GSE| >= 6 and |r| < 8 kpc
    bridge          everything else

Birth class is the paper's: halo-born = eps <= 0.8 AND z_max >= 1.5 kpc.  Masses
are GFM_InitialMass, i.e. mass at birth.

WHERE THIS STOPS MEANING ANYTHING.  Coalescence is at t = 5.4 Gyr.  After it the
debris phase-mixes, its centroid walks in towards the origin, and "within 6 kpc of
the centroid" stops picking out a satellite -- it picks out whatever happens to lie
near a drifting point.  The per-snapshot table below prints the centroid distance
and the surviving satellite stellar mass so that degradation is visible, and the
summary quotes the fraction both over the full window and over the pre-coalescence
part alone.  The difference between those two is the honest uncertainty on the
answer.

Writes out/bridge_mass_fraction.npz and prints the table.
"""
import gc, os, sys
import numpy as np
import auriga_public as ap
import config_au18 as C
import au18_frame as AF

TLO, THI = 4.9, 5.5
T_COAL = 5.4
CUT, ZCUT = 0.8, 1.5
R_SAT, R_HOST = 6.0, 8.0
OUT = C.OUT_DIR + '/bridge_mass_fraction.npz'

a = np.load(C.OUT_DIR + '/birth_orbits_actions.npz')
zx = np.load(C.OUT_DIR + '/birth_orbits_zmax.npz')
q = np.load(C.OUT_DIR + '/insitu_imass.npz')
st = np.load(C.OUT_DIR + '/snapshot_times.npz')
GSE = np.sort(np.load(C.OUT_DIR + '/gse_clean_ids.npy'))
assert np.array_equal(q['ids'], a['ids'])
IDS, TF, EPS, ZM, MI = a['ids'], a['tform'], a['eps_birth'], zx['zmax_birth'], q['imass']
SN_ALL, T_ALL = st['snaps'], st['t_snap']

# every snapshot whose (t_{k-1}, t_k] bin overlaps the window
ks = [k for k in range(1, len(SN_ALL))
      if T_ALL[k] > TLO and T_ALL[k - 1] < THI]
print(f'window {TLO}-{THI} Gyr spans snapshots '
      f'{", ".join(str(int(SN_ALL[k])) for k in ks)}')

SITES = ['GS/E analogue', 'bridge', 'MW analogue']
tot = {s: 0. for s in SITES}
tot_pre = {s: 0. for s in SITES}
rows = []
print(f'\n{"snap":>5s} {"t_form bin":>14s} {"d_GSE":>7s} {"M*_sat":>9s} '
      f'{"N_halo":>8s} {"M_halo":>10s} | ' +
      ' '.join(f'{s.split()[0]:>10s}' for s in SITES) + '   bridge %')
for k in ks:
    sn = int(SN_ALL[k])
    lo, hi = max(float(T_ALL[k - 1]), TLO), min(float(T_ALL[k]), THI)
    sel = (TF > lo) & (TF <= hi) & np.isfinite(EPS) & np.isfinite(ZM) & np.isfinite(MI)
    if sel.sum() == 0:
        continue
    sub = ap.subhalos.subfind(sn, directory=C.SIM_DIR,
                              loadlist=['SubhaloPos', 'Group_R_Crit200'])
    r200 = float(sub.data['Group_R_Crit200'][0]); cen = sub.data['SubhaloPos'][0]
    ref = ap.snapshot.load_snapshot(sn, 4, snappath=C.SIM_DIR, verbose=False,
        loadlist=['Coordinates', 'Masses', 'Potential', 'Velocities'])
    ref = ap.util.CentreOnHalo(ref, cen)
    ref = ap.util.apply_mask(ref, stars=False, radialcut=.5 * r200)
    ist, = np.where(ap.util.r(ref) < .1 * r200)
    L = np.cross(ref.data['Coordinates'][ist],
                 ref.data['Velocities'][ist] * ref.data['Masses'][ist, None])
    Ld = L.sum(0); Ld /= np.sqrt((Ld ** 2).sum())
    xd, yd, zd = ap.util.get_principal_axis(ref, ist, L=Ld)
    del ref; gc.collect()

    s = ap.snapshot.load_snapshot(sn, 4, snappath=C.SIM_DIR, verbose=False,
        loadlist=['Coordinates', 'Masses', 'ParticleIDs'])
    s = ap.util.CentreOnHalo(s, cen); ap.util.rotateto(s, xd, dir2=yd, dir3=zd)
    c = s.data['Coordinates']
    pos = np.column_stack([c[:, 1], c[:, 2], c[:, 0]]) * 1e3
    sid = s.data['ParticleIDs']
    smass = s.data['Masses'] * C.MASS_TO_MSUN
    del s, c; gc.collect()

    o = np.argsort(sid); ss = sid[o]
    p = np.searchsorted(ss, GSE)
    okg = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == GSE)
    Gp, Gm = pos[o[p[okg]]], smass[o[p[okg]]]
    gcen = np.median(Gp, axis=0)
    d_host_g = float(np.linalg.norm(gcen))
    Msat = float(Gm[np.linalg.norm(Gp - gcen, axis=1) < R_SAT].sum())

    ids_w = IDS[sel]
    pw = np.searchsorted(ss, ids_w)
    hit = (pw < len(ss)) & (ss[np.minimum(pw, len(ss) - 1)] == ids_w)
    Pw = pos[o[pw[hit]]]
    mw = MI[sel][hit]
    halo = ~((EPS[sel][hit] > CUT) | (ZM[sel][hit] < ZCUT))
    del pos, sid, smass; gc.collect()

    dg = np.linalg.norm(Pw - gcen, axis=1)
    dh = np.linalg.norm(Pw, axis=1)
    msk = {'GS/E analogue': dg < R_SAT,
           'MW analogue': (dg >= R_SAT) & (dh < R_HOST)}
    msk['bridge'] = ~(msk['GS/E analogue'] | msk['MW analogue'])
    Mh = mw[halo].sum()
    vals = {s_: float(mw[halo & msk[s_]].sum()) for s_ in SITES}
    for s_ in SITES:
        tot[s_] += vals[s_]
        if T_ALL[k] <= T_COAL:
            tot_pre[s_] += vals[s_]
    rows.append([sn, lo, hi, d_host_g, Msat, int(halo.sum()), Mh] +
                [vals[s_] for s_ in SITES])
    print(f'{sn:>5d} {f"{lo:.3f}-{hi:.3f}":>14s} {d_host_g:>7.1f} {Msat:>9.2e} '
          f'{halo.sum():>8,} {Mh:>10.3e} | ' +
          ' '.join(f'{vals[s_]:>10.3e}' for s_ in SITES) +
          f'   {100 * vals["bridge"] / Mh:>5.1f}%')
    del Pw, mw; gc.collect()

T = sum(tot.values())
Tp = sum(tot_pre.values())
print(f'\nTOTAL over {TLO}-{THI} Gyr, halo-born only: {T:.4e} Msun')
for s_ in SITES:
    print(f'  {s_:<16s} {tot[s_]:>10.4e}  {100 * tot[s_] / T:>5.1f} per cent')
print(f'\nPRE-COALESCENCE ONLY (snapshots ending at or before t = {T_COAL} Gyr): '
      f'{Tp:.4e} Msun')
for s_ in SITES:
    print(f'  {s_:<16s} {tot_pre[s_]:>10.4e}  {100 * tot_pre[s_] / Tp:>5.1f} per cent')
np.savez(OUT, rows=np.array(rows, float),
         cols=np.array(['snap', 't_lo', 't_hi', 'd_GSE', 'Msat', 'N_halo', 'M_halo',
                        'M_sat_site', 'M_bridge', 'M_host_site']))
print(f'\nsaved {OUT}')
