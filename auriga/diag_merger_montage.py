"""Diagnostic 1: spatial montage of the GS/E merger across snapshots.

Each panel shows the whole-galaxy stellar density (grey) with the tracked GS/E
debris overplotted (red), centred on the main halo, around the merger epoch.
Panels labelled with cosmic time, lookback time, and redshift.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod

os.makedirs(C.FIG_DIR, exist_ok=True)
gse_ids = np.sort(np.load(C.OUT_DIR + "/gse_clean_ids.npy"))

# snapshots spanning infall -> disruption -> settled
snap_list = [58, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 82]
W = 160.0    # half-window [kpc]

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for ax, sn in zip(axes.flat, snap_list):
    s = snap_mod.load_snapshot(sn, 4, snappath=C.SIM_DIR,
                               loadlist=["ParticleIDs", "Coordinates",
                                         "GFM_StellarFormationTime"])
    sid = s.data["ParticleIDs"]; coords = s.data["Coordinates"]
    real = s.data["GFM_StellarFormationTime"] > 0
    a = s.time; t = C.a_to_age(a); lb = C.T0_GYR - t; z = 1.0 / a - 1.0

    sf = sub_mod.subfind(sn, directory=C.SIM_DIR,
                         loadlist=["GroupFirstSub", "SubhaloPos", "Group_R_Crit200"])
    center = sf.data["SubhaloPos"][int(sf.data["GroupFirstSub"][0])]
    r200 = float(sf.data["Group_R_Crit200"][0]) * 1000.0

    rel = (coords - center) * 1000.0            # kpc, [Z,Y,X]
    x, y = rel[:, 2], rel[:, 1]                  # X, Y projection

    # background: all real stars
    m = real & (np.abs(x) < W) & (np.abs(y) < W)
    ax.hist2d(x[m], y[m], bins=200, range=[[-W, W], [-W, W]],
              cmap="Greys", cmin=1, norm="log")

    # GS/E debris present in this snapshot
    o = np.argsort(sid); ss = sid[o]
    pos = np.clip(np.searchsorted(ss, gse_ids), 0, len(ss) - 1)
    ok = ss[pos] == gse_ids
    gidx = o[pos][ok]
    gx, gy = rel[gidx, 2], rel[gidx, 1]
    ax.scatter(gx, gy, s=1.0, c="crimson", alpha=0.25, lw=0, rasterized=True)

    ax.add_patch(Circle((0, 0), r200, fill=False, ec="tab:blue", ls="--", lw=1))
    ax.plot(0, 0, "+", color="k", ms=8)
    ax.set_xlim(-W, W); ax.set_ylim(-W, W); ax.set_aspect("equal")
    ax.set_title(f"snap {sn}: t={t:.2f} Gyr, lookback={lb:.2f} Gyr, z={z:.2f}",
                 fontsize=10)
    ax.set_xticks([-100, 0, 100]); ax.set_yticks([-100, 0, 100])

fig.suptitle("Au18 GS/E merger — debris (red) infalling into the main galaxy; "
             "dashed circle = R200", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.98])
out = C.FIG_DIR + "/au18_gse_merger_montage.png"
fig.savefig(out, dpi=120); print("saved", out)
