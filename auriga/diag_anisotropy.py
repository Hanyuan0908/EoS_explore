"""Diagnostic 2: velocity anisotropy of the clean GS/E debris at z=0.

Disc-aligned galactocentric cylindrical velocities. GS/E should show the classic
radially-elongated 'sausage' in vR-vphi (little net rotation, large sigma_R) and
beta ~ 0.8-0.9, in contrast to the rotationally-supported in-situ disc.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod, util

os.makedirs(C.FIG_DIR, exist_ok=True)
gse_ids = np.sort(np.load(C.OUT_DIR + "/gse_clean_ids.npy"))

s = snap_mod.load_snapshot(127, 4, snappath=C.SIM_DIR,
                           loadlist=["ParticleIDs", "Coordinates", "Velocities",
                                     "GFM_StellarFormationTime", "GFM_InitialMass",
                                     "Masses", "GFM_Metals"])
real = s.data["GFM_StellarFormationTime"] > 0
# drop wind particles from every field so utilities operate on stars only
for k in list(s.data):
    s.data[k] = s.data[k][real]

sf = sub_mod.subfind(127, directory=C.SIM_DIR,
                     loadlist=["GroupFirstSub", "SubhaloPos"])
center = sf.data["SubhaloPos"][int(sf.data["GroupFirstSub"][0])]

# centre, remove bulk motion, align disc to z-axis (auriga_public utils)
util.CentreOnHalo(s, center)
# bulk (systemic) velocity = mass-weighted mean within 10 kpc
_rr = np.sqrt((s.data["Coordinates"] ** 2).sum(1))
_in = _rr < 0.01
bulk = np.average(s.data["Velocities"][_in], axis=0, weights=s.data["Masses"][_in])
s.data["Velocities"] -= bulk[None, :]
util.align_galaxy(s, radialcut=0.01)   # rotates Coordinates & Velocities in place

# align_galaxy puts the disc angular momentum along component 0, so the disc
# plane is components (1,2) and the rotation/symmetry axis is component 0.
co = s.data["Coordinates"] * 1000.0                         # kpc
ve = s.data["Velocities"]
zc = co[:, 0]; p, q = co[:, 1], co[:, 2]                    # zc = height along axis
vz = ve[:, 0]; vp, vq = ve[:, 1], ve[:, 2]
R = np.sqrt(p ** 2 + q ** 2)
good = R > 0.5
vR = np.where(good, (p * vp + q * vq) / np.where(good, R, 1), 0.0)
vphi = np.where(good, (p * vq - q * vp) / np.where(good, R, 1), 0.0)
rr = np.sqrt(zc ** 2 + p ** 2 + q ** 2)                     # kpc

sid = s.data["ParticleIDs"]
o = np.argsort(sid); ss = sid[o]
pos = np.clip(np.searchsorted(ss, gse_ids), 0, len(ss) - 1)
ok = ss[pos] == gse_ids
gidx = o[pos][ok]
gmask = np.zeros(len(sid), bool); gmask[gidx] = True

# make disc rotation positive
disc0 = (rr < 15) & (R > 3) & good
if np.median(vphi[disc0]) < 0:
    vphi = -vphi

def beta(m):
    sr, sp, sz = np.std(vR[m]), np.std(vphi[m]), np.std(vz[m])
    return 1 - (sp ** 2 + sz ** 2) / (2 * sr ** 2), sr, sp, sz

# samples: GS/E debris (inner halo) vs in-situ disc
gse_inner = gmask & (rr < 30) & good

bg, srg, spg, szg = beta(gse_inner)
bd, srd, spd, szd = beta((~gmask) & (rr < 20) & good)
print(f"GS/E debris (r<30kpc): N={gse_inner.sum()}  <vphi>={np.mean(vphi[gse_inner]):.0f}  "
      f"sigma_R={srg:.0f} sigma_phi={spg:.0f} sigma_z={szg:.0f}  beta={bg:.2f}")
print(f"in-situ (r<20kpc):     N={((~gmask)&(rr<20)&good).sum()}  <vphi>={np.mean(vphi[(~gmask)&(rr<20)&good]):.0f}  "
      f"sigma_R={srd:.0f} sigma_phi={spd:.0f} sigma_z={szd:.0f}  beta={bd:.2f}")

# spherical anisotropy (Fattahi+2019 convention): beta = 1 - sigma_t^2/(2 sigma_r^2)
vr_sph = (zc * vz + p * vp + q * vq) / np.where(rr > 0, rr, 1)   # spherical radial
# tangential speed^2 = |v|^2 - vr^2
vtot2 = vz ** 2 + vp ** 2 + vq ** 2
vt2 = np.clip(vtot2 - vr_sph ** 2, 0, None)

def beta_sph(m):
    sr2 = np.var(vr_sph[m])
    st2 = 0.5 * np.mean(vt2[m])          # sigma_theta^2+sigma_phi^2 ~ <vt^2> for <vt>~0
    return 1 - st2 / (2 * sr2)

print("\nGS/E debris beta(r)  [cylindrical / spherical]:")
for lo, hi in [(5, 15), (15, 25), (25, 40), (5, 40)]:
    m = gmask & good & (rr >= lo) & (rr < hi)
    if m.sum() > 50:
        b, sr, sp, sz = beta(m)
        print(f"  r={lo:2d}-{hi:2d} kpc: N={m.sum():5d}  beta_cyl={b:.2f}  "
              f"beta_sph={beta_sph(m):.2f}  (sR={sr:.0f} sp={sp:.0f} sz={sz:.0f})")
BETA_SPH_MAIN = beta_sph(gmask & good & (rr >= 5) & (rr < 40))

# ---- figure ----
fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2))
a0 = ax[0]
insb = (~gmask) & (rr < 20) & good
a0.hist2d(vphi[insb], vR[insb], bins=120, range=[[-350, 400], [-350, 350]],
          cmap="Blues", cmin=1, norm="log")
a0.scatter(vphi[gse_inner], vR[gse_inner], s=2, c="crimson", alpha=0.3, lw=0,
           rasterized=True, label="GS/E debris")
a0.axhline(0, color="gray", lw=0.6); a0.axvline(0, color="gray", lw=0.6)
a0.set_xlabel(r"$v_\phi$ [km/s]"); a0.set_ylabel(r"$v_R$ [km/s]")
a0.set_title(f"(a) $v_R$–$v_\\phi$: GS/E 'sausage' (red) vs in-situ disc (blue)")
a0.legend(loc="upper left", fontsize=9)
a0.text(0.03, 0.03,
        f"GS/E (r=5-40 kpc): $\\langle v_\\phi\\rangle$={np.mean(vphi[gse_inner]):.0f} km/s\n"
        f"$\\beta_{{sph}}$={BETA_SPH_MAIN:.2f}  ($\\beta_{{cyl}}$={bg:.2f})\n"
        f"$\\sigma_R$={srg:.0f}, $\\sigma_\\phi$={spg:.0f}, $\\sigma_z$={szg:.0f} km/s",
        transform=a0.transAxes, fontsize=8, va="bottom",
        bbox=dict(boxstyle="round", fc="white", alpha=0.85))

a1 = ax[1]
bins = np.linspace(-350, 350, 60)
a1.hist(vphi[gse_inner], bins=bins, density=True, histtype="step", color="crimson",
        lw=2, label="GS/E $v_\\phi$")
a1.hist(vR[gse_inner], bins=bins, density=True, histtype="step", color="k",
        lw=2, ls="--", label="GS/E $v_R$")
a1.hist(vphi[insb], bins=bins, density=True, histtype="step", color="tab:blue",
        lw=1.5, label="in-situ $v_\\phi$")
a1.set_xlabel("velocity [km/s]"); a1.set_ylabel("normalised")
a1.set_title("(b) velocity distributions")
a1.legend(fontsize=9)

fig.suptitle("Au18 GS/E debris kinematics at z=0 — radially anisotropic, non-rotating",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = C.FIG_DIR + "/au18_gse_anisotropy.png"
fig.savefig(out, dpi=140); print("saved", out)
