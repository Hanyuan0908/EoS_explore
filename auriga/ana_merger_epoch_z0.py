"""Present-day chemistry and kinematics of stars born around the Au18 merger.

Membership is deliberately conservative: host cohorts use only IDs classified
as in-situ by the assembled provenance catalogue, while the merger reference is
the clean GS/E ID list. Thus stars born in unrelated satellites are excluded.
The merger window is the union of the birth intervals displayed for snapshots
73--82; equal-duration adjacent windows provide before/after controls.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod, util


os.makedirs(C.FIG_DIR, exist_ok=True)
os.makedirs(C.OUT_DIR, exist_ok=True)

matched = np.load(C.OUT_DIR + "/matched_z0.npz")
insitu_idx = matched["ii"]
gse_ids = np.sort(np.load(C.OUT_DIR + "/gse_clean_ids.npy"))

s = snap_mod.load_snapshot(
    127, 4, snappath=C.SIM_DIR,
    loadlist=["ParticleIDs", "Coordinates", "Velocities", "Masses",
              "GFM_StellarFormationTime", "GFM_InitialMass", "GFM_Metals"],
)
real = s.data["GFM_StellarFormationTime"] > 0
original_idx = np.flatnonzero(real)
old_to_real = np.full(len(real), -1, dtype=np.int64)
old_to_real[original_idx] = np.arange(len(original_idx))
insitu_real_idx = old_to_real[insitu_idx]
insitu_real_idx = insitu_real_idx[insitu_real_idx >= 0]

for key in list(s.data):
    s.data[key] = s.data[key][real]

sf = sub_mod.subfind(127, directory=C.SIM_DIR,
                     loadlist=["GroupFirstSub", "SubhaloPos"])
center = sf.data["SubhaloPos"][int(sf.data["GroupFirstSub"][0])]
util.CentreOnHalo(s, center)
radius_mpc = np.sqrt(np.sum(s.data["Coordinates"] ** 2, axis=1))
inner10 = radius_mpc < 0.01
bulk = np.average(s.data["Velocities"][inner10], axis=0,
                  weights=s.data["Masses"][inner10])
s.data["Velocities"] -= bulk[None, :]
util.align_galaxy(s, radialcut=0.01)

# align_galaxy puts the disc axis on component 0.
xyz = s.data["Coordinates"] * 1000.0
vel = s.data["Velocities"]
z, p, q = xyz[:, 0], xyz[:, 1], xyz[:, 2]
vz, vp, vq = vel[:, 0], vel[:, 1], vel[:, 2]
R = np.hypot(p, q)
r = np.sqrt(z*z + R*R)
good = R > 0.5
vR = (p*vp + q*vq) / np.where(good, R, 1.0)
vphi = (p*vq - q*vp) / np.where(good, R, 1.0)
if np.median(vphi[(r < 15) & (R > 3)]) < 0:
    vphi *= -1

sid = s.data["ParticleIDs"]
order = np.argsort(sid)
sorted_sid = sid[order]
pos = np.searchsorted(sorted_sid, gse_ids)
valid = pos < len(sorted_sid)
gidx = order[pos[valid & (sorted_sid[np.minimum(pos, len(sorted_sid)-1)] == gse_ids)]]
gse = np.zeros(len(sid), dtype=bool)
gse[gidx] = True
insitu = np.zeros(len(sid), dtype=bool)
insitu[insitu_real_idx] = True

birth = C.a_to_age(s.data["GFM_StellarFormationTime"])
feh = C.bracket_abundance(s.data["GFM_Metals"], "Fe", "H")
mgfe = C.bracket_abundance(s.data["GFM_Metals"], "Mg", "Fe")
imass = s.data["GFM_InitialMass"] * C.MASS_TO_MSUN

# Panel 73 contains stars formed after snapshot 72; panel 82 ends at snapshot 82.
t_start = float(C.a_to_age(snap_mod.load_snapshot(
    72, 4, snappath=C.SIM_DIR, loadlist=["GFM_StellarFormationTime"]
).time))
t_end = float(C.a_to_age(snap_mod.load_snapshot(
    82, 4, snappath=C.SIM_DIR, loadlist=["GFM_StellarFormationTime"]
).time))
dt = t_end - t_start

# Same present-day aperture for every population.
aperture = good & (r >= 3) & (r < 30) & np.isfinite(feh) & np.isfinite(mgfe)
masks = {
    "Clean GS/E": gse & aperture,
    "Host: before": insitu & aperture & (birth >= t_start-dt) & (birth < t_start),
    "Host: during": insitu & aperture & (birth >= t_start) & (birth <= t_end),
    "Host: after": insitu & aperture & (birth > t_end) & (birth <= t_end+dt),
}
colors = ["crimson", "#7b3294", "#e66101", "#018571"]

print(f"birth windows [Gyr]: before={t_start-dt:.2f}-{t_start:.2f}, "
      f"during={t_start:.2f}-{t_end:.2f}, after={t_end:.2f}-{t_end+dt:.2f}")
print("sample             N     Minit       <vphi> sigR sigphi sigz  beta  medFeH medMgFe")
rows = []
for (name, m), color in zip(masks.items(), colors):
    sr, sp, sz = np.std(vR[m]), np.std(vphi[m]), np.std(vz[m])
    beta = 1.0 - (sp*sp + sz*sz)/(2.0*sr*sr)
    row = (name, m.sum(), imass[m].sum(), np.mean(vphi[m]), sr, sp, sz, beta,
           np.median(feh[m]), np.median(mgfe[m]))
    rows.append(row)
    print(f"{name:16s} {row[1]:7d} {row[2]:.3e} {row[3]:7.1f} "
          f"{sr:4.0f} {sp:6.0f} {sz:4.0f} {beta:5.2f} "
          f"{row[8]:+6.2f} {row[9]:+7.2f}")

fig, axes = plt.subplots(2, 4, figsize=(16, 8.1), sharex="row", sharey="row")
for col, ((name, m), color) in enumerate(zip(masks.items(), colors)):
    ax = axes[0, col]
    ax.hexbin(feh[m], mgfe[m], gridsize=65, extent=(-2.5, 0.7, -0.25, 0.65),
              bins="log", mincnt=1, cmap="magma")
    ax.axhline(0, color="0.7", lw=0.5)
    ax.axvline(0, color="0.7", lw=0.5)
    ax.set_title(f"{name}\nN={m.sum():,}", color=color, fontsize=11)
    ax.set_xlabel("[Fe/H]")
    if col == 0:
        ax.set_ylabel("[Mg/Fe]")

    ax = axes[1, col]
    ax.hexbin(vphi[m], vR[m], gridsize=65, extent=(-350, 400, -350, 350),
              bins="log", mincnt=1, cmap="viridis")
    ax.axhline(0, color="0.7", lw=0.5)
    ax.axvline(0, color="0.7", lw=0.5)
    ax.set_xlabel(r"$v_\phi$ [km s$^{-1}$]")
    if col == 0:
        ax.set_ylabel(r"$v_R$ [km s$^{-1}$]")
    row = rows[col]
    ax.text(0.03, 0.97, f"<vphi>={row[3]:.0f} km/s\nbeta={row[7]:.2f}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(fc="white", alpha=0.75, ec="none"))

fig.suptitle(
    f"Au18 at z=0: clean GS/E and host-born cohorts around the merger "
    f"(birth t={t_start:.2f}-{t_end:.2f} Gyr; 3<r<30 kpc)", fontsize=13,
)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig_out = C.FIG_DIR + "/au18_merger_epoch_z0_chemo_kinematics.png"
fig.savefig(fig_out, dpi=140)

np.savez(
    C.OUT_DIR + "/merger_epoch_z0_samples.npz",
    t_start=t_start, t_end=t_end,
    **{name.lower().replace(" ", "_").replace(":", ""): sid[m]
       for name, m in masks.items()},
)
print("saved", fig_out)
