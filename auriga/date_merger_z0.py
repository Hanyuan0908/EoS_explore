"""Step 1+2: z=0 characterisation for dating the Au18 GS/E merger.

- Read the z=0 accreted-star provenance file directly (unambiguous membership).
- Group ex-situ stars by RootIndex -> distinct accreted progenitors.
- Characterise each from SNAPSHOT quantities (stellar mass, [Fe/H], birth time,
  present-day radius) to identify the GS/E (most massive accreted progenitor)
  and read off its star-formation truncation (upper bound on the merger time).
- Build birth-time (SFR) histories: GS/E ex-situ stars vs all in-situ main-galaxy
  stars (the merger-induced starburst).

Outputs arrays + a summary to auriga/out/ for the next steps.
"""
import os
import numpy as np
import h5py

import config_au18 as C
from auriga_public import snapshot as snap_mod
from auriga_public import subhalos as sub_mod

os.makedirs(C.OUT_DIR, exist_ok=True)
SNAP = 127

# --------------------------------------------------------------------------- #
# 1. z=0 provenance (single file, direct read)
# --------------------------------------------------------------------------- #
prov_file = os.path.join(C.ACCRETED_DIR, C.HALO,
                         f"{C.HALO}starID_accreted_all_newmtree_{SNAP:03d}.hdf5")
with h5py.File(prov_file, "r") as f:
    ex_ids = f["Exsitu/ParticleIDs"][:]
    ex_root = f["Exsitu/RootIndex"][:]
    ex_peakidx = f["Exsitu/PeakMassIndex"][:]
    ex_peakinf = f["Exsitu/PeakMassInfalltime"][:]
    ins_ids = f["Insitu/ParticleIDs"][:]
print(f"z=0 provenance: {len(ex_ids)} ex-situ, {len(ins_ids)} in-situ (halo catalogue)")

# --------------------------------------------------------------------------- #
# 2. z=0 snapshot star particles
# --------------------------------------------------------------------------- #
s = snap_mod.load_snapshot(SNAP, 4, snappath=C.SIM_DIR,
                           loadlist=["ParticleIDs", "Coordinates", "Velocities",
                                     "GFM_StellarFormationTime", "GFM_InitialMass",
                                     "Masses", "GFM_Metallicity", "GFM_Metals"])
sid = s.data["ParticleIDs"]
aform = s.data["GFM_StellarFormationTime"]
real = aform > 0                      # drop wind particles
print(f"snapshot PartType4: {len(sid)} entries, {real.sum()} real stars")

coords = s.data["Coordinates"]        # physical kpc, [Z,Y,X]
imass = s.data["GFM_InitialMass"] * C.MASS_TO_MSUN   # Msun
metals = s.data["GFM_Metals"]

# galaxy centre = central subhalo of FoF0
sf = sub_mod.subfind(SNAP, directory=C.SIM_DIR,
                     loadlist=["GroupPos", "GroupFirstSub", "Group_R_Crit200",
                               "SubhaloPos", "SubhaloMassType", "SubhaloLenType"])
cen_sub = int(sf.data["GroupFirstSub"][0])
center = sf.data["SubhaloPos"][cen_sub]
r200 = float(sf.data["Group_R_Crit200"][0])
print(f"centre = {center}  R200 = {r200:.1f} kpc")

rad = np.sqrt(((coords - center) ** 2).sum(axis=1))

feh = C.bracket_abundance(metals, "Fe", "H")
mgfe = C.bracket_abundance(metals, "Mg", "Fe")
age = C.a_to_age(aform)               # cosmic time of birth [Gyr]

# --------------------------------------------------------------------------- #
# 3. match ex-situ IDs -> snapshot rows
# --------------------------------------------------------------------------- #
order = np.argsort(sid)
sid_sorted = sid[order]

def match(ids):
    pos = np.searchsorted(sid_sorted, ids)
    pos = np.clip(pos, 0, len(sid_sorted) - 1)
    ok = sid_sorted[pos] == ids
    return order[pos[ok]], ok

ex_idx, ex_ok = match(ex_ids)
ex_root_m = ex_root[ex_ok]
print(f"matched {ex_ok.sum()}/{len(ex_ids)} ex-situ IDs to snapshot")

# --------------------------------------------------------------------------- #
# 4. characterise accreted progenitors by RootIndex
# --------------------------------------------------------------------------- #
roots, counts = np.unique(ex_root_m, return_counts=True)
odr = np.argsort(-counts)
print("\n== accreted progenitors (RootIndex) by stellar mass ==")
print(" root      N   Mstar[Msun]  med[Fe/H]  medBirthAge  medRad  peakInf")
rows = []
for k in odr[:12]:
    rk = roots[k]
    sel = ex_idx[ex_root_m == rk]
    m = imass[sel].sum()
    fk = np.nanmedian(feh[sel])
    bk = np.nanmedian(age[sel])
    rk_rad = np.nanmedian(rad[sel])
    pinf = np.nanmedian(ex_peakinf[ex_ok][ex_root_m == rk])
    rows.append((rk, len(sel), m, fk, bk, rk_rad, pinf))
    print(f" {rk:5d} {len(sel):6d}  {m:.3e}   {fk:+.2f}     {bk:6.2f}    "
          f"{rk_rad:6.1f}  {pinf:6.2f}")

# GS/E = most massive accreted progenitor (by stellar mass)
gse_root = rows[int(np.argmax([row[2] for row in rows]))][0]
gse_sel = ex_idx[ex_root_m == gse_root]
print(f"\n>>> GS/E candidate = RootIndex {gse_root}: "
      f"N={len(gse_sel)}, Mstar={imass[gse_sel].sum():.3e} Msun, "
      f"median[Fe/H]={np.nanmedian(feh[gse_sel]):+.2f}")
gse_ids = sid[gse_sel]

# --------------------------------------------------------------------------- #
# 5. in-situ main-galaxy stars (SFR history)
# --------------------------------------------------------------------------- #
ex_id_set = np.zeros(len(sid), bool); ex_id_set[ex_idx] = True
gal = real & (rad < 0.15 * r200)      # main galaxy: inside 0.15 R200
insitu_gal = gal & (~ex_id_set)
print(f"\nmain galaxy (r<0.15R200): {gal.sum()} stars, "
      f"{insitu_gal.sum()} in-situ, {(gal & ex_id_set).sum()} ex-situ")

# --------------------------------------------------------------------------- #
# 6. save
# --------------------------------------------------------------------------- #
np.savez(os.path.join(C.OUT_DIR, "z0_dating.npz"),
         gse_root=gse_root, gse_ids=gse_ids,
         gse_birth_age=age[gse_sel], gse_feh=feh[gse_sel], gse_mgfe=mgfe[gse_sel],
         gse_rad=rad[gse_sel], gse_aform=aform[gse_sel], gse_imass=imass[gse_sel],
         insitu_birth_age=age[insitu_gal], insitu_feh=feh[insitu_gal],
         insitu_mgfe=mgfe[insitu_gal], insitu_imass=imass[insitu_gal],
         insitu_aform=aform[insitu_gal],
         center=center, r200=r200)
print(f"\nsaved -> {os.path.join(C.OUT_DIR, 'z0_dating.npz')}")

# quick SF-truncation readout for GS/E
ga = age[gse_sel]
print(f"\nGS/E star-formation: birth ages 5-95pct = "
      f"{np.nanpercentile(ga,5):.2f} - {np.nanpercentile(ga,95):.2f} Gyr; "
      f"last 10% form after {np.nanpercentile(ga,90):.2f} Gyr "
      f"(lookback {C.T0_GYR-np.nanpercentile(ga,90):.2f})")
