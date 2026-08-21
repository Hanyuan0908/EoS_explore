"""Step 3: orbital-decay track of the GS/E progenitor across snapshots.

Follow the z=0-defined GS/E debris star IDs back through time. While the
satellite is intact its stars form a compact clump orbiting the main halo;
at coalescence the clump's galactocentric distance decays to ~0 and its
spatial dispersion inflates (phase mixing). That transition dates the merger.

Usage: python date_merger_track.py <snap_start> <snap_end> <step>
Writes auriga/out/gse_track_<start>_<end>_<step>.npz
"""
import sys, os
import numpy as np
import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod

start = int(sys.argv[1]) if len(sys.argv) > 1 else 55
end = int(sys.argv[2]) if len(sys.argv) > 2 else 127
step = int(sys.argv[3]) if len(sys.argv) > 3 else 2
ids_file = sys.argv[4] if len(sys.argv) > 4 else "gse_proxy_ids.npy"
tag = sys.argv[5] if len(sys.argv) > 5 else "proxy"

gse_ids = np.load(C.OUT_DIR + "/" + ids_file)
gse_sorted = np.sort(gse_ids)
print(f"tracking {len(gse_ids)} GS/E [{tag}] stars over snaps {start}..{end} step {step}")

snaps, times, nfound = [], [], []
r_med, r_p25, r_p75, disp, frac20 = [], [], [], [], []

for sn in range(start, end + 1, step):
    try:
        s = snap_mod.load_snapshot(sn, 4, snappath=C.SIM_DIR,
                                   loadlist=["ParticleIDs", "Coordinates"],
                                   verbose=False)
    except Exception as e:
        print(f"  snap {sn}: load failed ({e})"); continue
    sid = s.data["ParticleIDs"]; coords = s.data["Coordinates"]  # physical Mpc
    a = s.time

    # main-halo centre
    sf = sub_mod.subfind(sn, directory=C.SIM_DIR,
                         loadlist=["GroupFirstSub", "SubhaloPos", "Group_R_Crit200"])
    center = sf.data["SubhaloPos"][int(sf.data["GroupFirstSub"][0])]

    # locate proxy stars present in this snapshot
    o = np.argsort(sid); ss = sid[o]
    pos = np.clip(np.searchsorted(ss, gse_sorted), 0, len(ss) - 1)
    ok = ss[pos] == gse_sorted
    idx = o[pos][ok]
    if ok.sum() < 20:
        print(f"  snap {sn}: only {ok.sum()} proxy stars present, skip")
        continue
    rel = (coords[idx] - center) * 1000.0            # kpc, relative to centre
    d = np.sqrt((rel ** 2).sum(1))

    # satellite clump: use the densest 50% (robust to already-stripped stars)
    # centre-of-clump = median position; dispersion about it
    clump_center = np.median(rel, axis=0)
    dclump = np.sqrt(((rel - clump_center) ** 2).sum(1))

    snaps.append(sn); times.append(C.a_to_age(a)); nfound.append(int(ok.sum()))
    r_med.append(np.median(d)); r_p25.append(np.percentile(d, 25))
    r_p75.append(np.percentile(d, 75))
    disp.append(np.median(dclump))                    # MAD-like clump size
    frac20.append(float((d < 20).mean()))
    print(f"  snap {sn}  t={C.a_to_age(a):5.2f} Gyr  N={ok.sum():6d}  "
          f"r_med={np.median(d):7.1f} kpc  clumpDisp={np.median(dclump):7.1f}  "
          f"f(<20kpc)={frac20[-1]:.2f}")

out = C.OUT_DIR + f"/gse_track_{tag}_{start}_{end}_{step}.npz"
np.savez(out, snaps=np.array(snaps), times=np.array(times), nfound=np.array(nfound),
         r_med=np.array(r_med), r_p25=np.array(r_p25), r_p75=np.array(r_p75),
         disp=np.array(disp), frac20=np.array(frac20))
print("saved", out)
