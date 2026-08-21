"""Gas and newly formed-star montage across the Au18 GS/E merger.

Gas surface density is shown in grey. Red points are real star particles whose
GFM_StellarFormationTime lies between the immediately preceding snapshot and
the displayed snapshot. This formation-time selection is preferable to ID
differencing because wind particles can change particle type.
"""
import os

import gc
import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle

import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod


os.makedirs(C.FIG_DIR, exist_ok=True)

# Match diag_merger_montage.py for direct visual comparison.
SNAP_LIST = [58, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 82]
W = 160.0  # half-width [kpc]


def snapshot_scale_factor(sn):
    """Read only the small HDF5 header, avoiding a second full snapshot load."""
    pattern = f"{C.SIM_DIR}/snapdir_{sn:03d}/snapshot_{sn:03d}.*.hdf5"
    filename = sorted(glob.glob(pattern))[0]
    with h5py.File(filename, "r") as handle:
        return float(handle["Header"].attrs["Time"])


fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for ax, sn in zip(axes.flat, SNAP_LIST):
    gas = snap_mod.load_snapshot(
        sn, 0, snappath=C.SIM_DIR,
        loadlist=["Coordinates", "Masses"],
    )
    stars = snap_mod.load_snapshot(
        sn, 4, snappath=C.SIM_DIR,
        loadlist=["Coordinates", "GFM_StellarFormationTime", "GFM_InitialMass"],
    )
    a = float(stars.time)
    a_prev = snapshot_scale_factor(sn - 1)
    t = float(C.a_to_age(a))
    t_prev = float(C.a_to_age(a_prev))
    lb = C.T0_GYR - t
    z = 1.0 / a - 1.0

    sf = sub_mod.subfind(
        sn, directory=C.SIM_DIR,
        loadlist=["GroupFirstSub", "SubhaloPos", "Group_R_Crit200"],
    )
    center = sf.data["SubhaloPos"][int(sf.data["GroupFirstSub"][0])]
    r200 = float(sf.data["Group_R_Crit200"][0]) * 1000.0

    grel = (gas.data["Coordinates"] - center) * 1000.0
    gx, gy = grel[:, 2], grel[:, 1]  # raw vectors are [Z,Y,X]
    ingas = (np.abs(gx) < W) & (np.abs(gy) < W)
    gas_mass = gas.data["Masses"] * C.MASS_TO_MSUN
    ax.hist2d(
        gx[ingas], gy[ingas], weights=gas_mass[ingas], bins=200,
        range=[[-W, W], [-W, W]], cmap="Greys", cmin=1.0,
        norm=LogNorm(),
    )

    aform = stars.data["GFM_StellarFormationTime"]
    newborn = (aform > a_prev) & (aform <= a)
    srel = (stars.data["Coordinates"] - center) * 1000.0
    sx, sy = srel[:, 2], srel[:, 1]
    inframe = newborn & (np.abs(sx) < W) & (np.abs(sy) < W)
    ax.scatter(
        sx[inframe], sy[inframe], s=2.0, c="crimson", alpha=0.65,
        lw=0, rasterized=True,
    )

    # Report only the population actually shown. This also excludes distant
    # low-resolution boundary particles with very large initial masses.
    n_new = np.count_nonzero(inframe)
    formed_mass = np.sum(stars.data["GFM_InitialMass"][inframe]) * C.MASS_TO_MSUN
    print(
        f"snap {sn:3d}: dt={t-t_prev:.3f} Gyr, newborn={n_new:,}, "
        f"Minit={formed_mass:.3e} Msun",
        flush=True,
    )

    ax.add_patch(Circle((0, 0), r200, fill=False, ec="tab:blue", ls="--", lw=1))
    ax.plot(0, 0, "+", color="k", ms=8)
    ax.text(
        0.02, 0.98,
        f"new stars: {n_new:,}\nformed mass: {formed_mass/1e9:.2f}e9 Msun",
        transform=ax.transAxes, va="top", ha="left", fontsize=8,
        bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=2),
    )
    ax.set_xlim(-W, W)
    ax.set_ylim(-W, W)
    ax.set_aspect("equal")
    ax.set_title(
        f"snap {sn}: t={t:.2f} Gyr, lookback={lb:.2f} Gyr, z={z:.2f}",
        fontsize=10,
    )
    ax.set_xticks([-100, 0, 100])
    ax.set_yticks([-100, 0, 100])
    del gas, stars, sf, grel, srel, gas_mass
    gc.collect()

fig.suptitle(
    "Au18 GS/E merger — gas surface density (grey) and stars born since the "
    "previous snapshot (red); dashed circle = R200",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 1, 0.98])
out = C.FIG_DIR + "/au18_gas_newstars_montage.png"
fig.savefig(out, dpi=120)
print("saved", out)
