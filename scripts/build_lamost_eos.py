"""Phase D: build a LAMOST sample with [Al/Fe] + precise MSTO ages + orbits.

Join DD-Payne abundances (MG_FE, AL_FE, FEH) onto the subgiant age catalogue by
SPECID, then apply the same chemical+kinematic Eos / low-alpha-disc / Splash
selection used for APOGEE.  Writes results/lamost_eos_sample.fits and prints
diagnostic counts.  Also validates against the APOGEE source_id cross-match.
"""
import _bootstrap  # noqa
import numpy as np
from astropy.table import Table, join

from eos import loaders, selections as S, config as C


def build(which="fullparam"):
    sg = loaders.load_lamost_subgiants(which)
    dp = loaders.load_ddpayne()

    # DD-Payne carries the abundances; subgiants carry ages+orbits. Join on specid.
    # DD-Payne can have multiple visits per specid string -> keep first occurrence.
    _, uniq = np.unique(dp["specid"], return_index=True)
    dp_u = dp[np.sort(uniq)]

    sg_t = Table({"specid": sg["specid"], "feh_sg": sg["feh"],
                  "alphafe": sg["alphafe"], "age": sg["age"], "age_err": sg["age_err"],
                  "ecc": sg["ecc"], "rap": sg["rap"], "rperi": sg["rperi"],
                  "Lz3": sg["Lz3"], "E5": sg["E5"], "vtan": sg["vtan"],
                  "source_id": sg["source_id"]})
    dp_t = Table({"specid": dp_u["specid"], "feh": dp_u["feh_dp"],
                  "mgfe": dp_u["mgfe"], "alfe": dp_u["alfe"],
                  "alfe_err": np.asarray(dp_u["AL_FE_ERR"], float),
                  "mg_flag": np.asarray(dp_u["FLAG_MG_FE"]),
                  "al_flag": np.asarray(dp_u["FLAG_AL_FE"])})

    m = join(sg_t, dp_t, keys="specid", join_type="inner")
    return m


def report(m, label):
    feh = np.asarray(m["feh"], float)
    print(f"\n[{label}] joined rows: {len(m)}")
    cnt = S.summary_counts(m)
    for k, v in cnt.items():
        print(f"   {k:14s}: {v}")
    eos = m[S.eos_mask(m)]
    rel = np.asarray(eos["age_err"]) / np.asarray(eos["age"])
    print(f"   Eos: age med={np.nanmedian(eos['age']):.2f} Gyr  "
          f"rel-age-err med={np.nanmedian(rel):.2f}  "
          f"feh med={np.nanmedian(np.asarray(eos['feh'])):.2f}  "
          f"ecc med={np.nanmedian(np.asarray(eos['ecc'])):.2f}")
    return eos


if __name__ == "__main__":
    m = build("fullparam")
    report(m, "DD-Payne x subgiant_fullparam")
    out = loaders.repo_path(C.RESULTS_DIR, "lamost_eos_sample.fits")
    m.write(out, overwrite=True)
    print(f"\nwrote {out}")

    # ---- Validation against APOGEE clean-Eos via source_id ----
    ap = loaders.load_apogee()
    ap_eos_sid = set(np.asarray(ap["source_id"])[S.eos_mask(ap)].tolist())
    lam_sid = np.asarray(m["source_id"])
    lam_is_eos = S.eos_mask(m)
    in_ap_eos = np.array([s in ap_eos_sid for s in lam_sid])
    overlap = int((in_ap_eos & lam_is_eos).sum())
    print(f"\nValidation: APOGEE-Eos stars also present in LAMOST sample: {int(in_ap_eos.sum())}")
    print(f"   of those, LAMOST chemistry+kinematics also flags Eos: {overlap}")
