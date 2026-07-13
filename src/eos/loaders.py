"""FITS loaders that return tidy astropy Tables with derived/standardised columns.

Each loader adds a common set of columns so downstream selection/plotting code is
catalogue-agnostic:
    feh, mgfe, alfe   -- abundances ([X/Fe], dex)
    E5, Lz3           -- energy & ang. mom. in the paper's scaled units
    vtan              -- galactocentric tangential (azimuthal) velocity [km/s]
    ecc, rap, rperi   -- orbit shape
    age, age_err      -- where available
    source_id         -- Gaia DR3 source id (int64) for cross-matching
"""
from __future__ import annotations

import os
import numpy as np
from astropy.table import Table

from . import config as C


def repo_path(*parts: str) -> str:
    """Path relative to the repo root (this file is src/eos/loaders.py)."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    p = os.path.join(root, *parts)
    os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    return p


def _to_kpc(dist):
    """astroNN weighted_dist is in kpc already in this VAC; guard either way."""
    d = np.asarray(dist, float)
    med = np.nanmedian(d)
    return d / 1000.0 if med > 100 else d  # pc -> kpc if values look like pc


def load_apogee(distance_cut: bool = True, clean: bool = True) -> Table:
    """APOGEE DR17 base sample, matching the paper's data choices:

    * chemistry  -> ASPCAP allStar (APOGEE_DR17_all.fits); abundances are already
      [X/Fe] / [Fe/H].  (The AstroNN *neural-net* abundances show a spurious gap
      near [Fe/H]~-0.55 and are NOT used for chemistry.)
    * orbits/kinematics/ages -> AstroNN VAC (MWPotential2014, galpy).

    The two catalogues are row-matched (same APOGEE_ID ordering).  Cleaning
    removes fill values, unreliable element measurements, STAR_BAD spectra, low
    S/N, non-giants, and duplicates/commissioning (EXTRATARG!=0) -- the paper's
    "clean of duplicates and unreliable measurements / red giants" step.
    """
    nn = Table.read(C.APOGEE_ASTRONN, hdu=1)
    asp = Table.read(C.APOGEE_ALLSTAR, hdu=1)
    assert len(nn) == len(asp), "allStar and AstroNN are not row-matched"

    t = Table()
    t["apogee_id"] = _norm_specid(asp["APOGEE_ID"])
    t["snr"] = np.asarray(asp["SNR"], float)
    # --- chemistry from ASPCAP (already [X/Fe]) ---
    t["feh"] = np.asarray(asp["FE_H"], float)
    t["mgfe"] = np.asarray(asp["MG_FE"], float)
    t["alfe"] = np.asarray(asp["AL_FE"], float)
    # --- orbits / kinematics / ages from AstroNN ---
    t["E5"] = np.asarray(nn["Energy"], float) * C.E_SCALE
    t["Lz"] = np.asarray(nn["Lz"], float)
    t["Lz3"] = t["Lz"] * C.LZ_SCALE
    t["vtan"] = np.asarray(nn["galvt"], float)
    t["ecc"] = np.asarray(nn["e"], float)
    t["rap"] = np.asarray(nn["rap"], float)
    t["rperi"] = np.asarray(nn["rperi"], float)
    t["age"] = np.asarray(nn["age_lowess_correct"], float)
    t["age_err"] = np.asarray(nn["age_total_error"], float)
    t["source_id"] = np.asarray(nn["source_id"], np.int64)
    t["dist_kpc"] = _to_kpc(nn["weighted_dist"])

    keep = np.ones(len(t), bool)
    if clean:
        for c in ("feh", "mgfe", "alfe"):
            keep &= np.asarray(t[c]) > -100        # drop -9999 fill
        for f in ("FE_H_FLAG", "MG_FE_FLAG", "AL_FE_FLAG"):
            keep &= np.asarray(asp[f]) == 0        # reliable element measurements
        af = np.asarray(asp["ASPCAPFLAG"], np.int64)
        keep &= (af & (1 << C.ASPCAP_STARBAD_BIT)) == 0
        keep &= np.asarray(asp["SNR"], float) > C.SNR_MIN
        logg = np.asarray(asp["LOGG"], float)
        keep &= (logg > C.LOGG_GIANT[0]) & (logg < C.LOGG_GIANT[1])
    if distance_cut:
        keep &= np.isfinite(np.asarray(t["dist_kpc"])) & (np.asarray(t["dist_kpc"]) < C.HELIO_DIST_MAX)
    # always require the chemistry the selection depends on
    keep &= np.isfinite(np.asarray(t["feh"])) & np.isfinite(np.asarray(t["mgfe"])) & np.isfinite(np.asarray(t["alfe"]))
    t = t[keep]

    if clean:
        t = _dedupe_by(t, "apogee_id", "snr")   # remove duplicate visits, keep best S/N
    return t


def load_lamost_subgiants(which: str = "fullparam") -> Table:
    """LAMOST subgiant age catalogues. Provides precise MSTO `age`.

    `which`: 'fullparam' (subgiant_fullparam_update) or 'xiang' (Xiang+2024).
    These have [alpha/Fe] but NOT [Al/Fe]; Al is added later via the DD-Payne join.
    """
    if which == "fullparam":
        t = Table.read(C.LAMOST_SUBGIANT, hdu=1)
        t["feh"] = np.asarray(t["FEH"], float)
        t["alphafe"] = np.asarray(t["ALPHA_FE"], float)
        t["age"] = np.asarray(t["AGE"], float)
        t["age_err"] = np.asarray(t["AGE_ERR"], float)
        t["ecc"] = np.asarray(t["ECC"], float)
        t["rap"] = np.asarray(t["R_APO"], float)
        t["rperi"] = np.asarray(t["R_PERI"], float)
        t["Lz3"] = np.asarray(t["LZ"], float) * C.LZ_SCALE
        t["E5"] = np.asarray(t["E"], float) * C.E_SCALE
        t["vtan"] = np.asarray(t["VT"], float)
        t["specid"] = _norm_specid(t["SPECID"])
        t["source_id"] = np.asarray(t["SOURCE_ID"], np.int64)
    elif which == "xiang":
        t = Table.read(C.LAMOST_SUBGIANT_XIANG, hdu=1)
        t["feh"] = np.asarray(t["FEH"], float)
        t["alphafe"] = np.asarray(t["ALPHA_FE"], float)
        t["age"] = np.asarray(t["AGE"], float)
        t["age_err"] = np.asarray(t["E_AGE"], float)
        t["ecc"] = np.asarray(t["ECC"], float)
        t["rap"] = np.asarray(t["R_APO"], float)
        t["rperi"] = np.asarray(t["R_PERI"], float)
        t["Lz3"] = np.asarray(t["LZ"], float) * C.LZ_SCALE
        t["E5"] = np.asarray(t["ENERGY"], float) * C.E_SCALE
        t["vtan"] = np.asarray(t["VT"], float)
        t["specid"] = _norm_specid(t["LAMOST_SPECID"])
        t["source_id"] = np.asarray(t["GAIAEDR3_SOURCE_ID"], np.int64)
    else:
        raise ValueError(which)
    return t


def load_ddpayne(columns=None) -> Table:
    """LAMOST DR9 DD-Payne abundances (9.2M rows). [X/Fe] already.

    Heavy file; read only what is needed.
    """
    cols = columns or ["SPECID", "FEH", "MG_FE", "AL_FE",
                       "MG_FE_ERR", "AL_FE_ERR", "FLAG_MG_FE", "FLAG_AL_FE"]
    t = Table.read(C.LAMOST_DDPAYNE, hdu=1)
    t = t[cols]
    t["specid"] = _norm_specid(t["SPECID"])
    t["feh_dp"] = np.asarray(t["FEH"], float)
    t["mgfe"] = np.asarray(t["MG_FE"], float)
    t["alfe"] = np.asarray(t["AL_FE"], float)
    return t


def _dedupe_by(t, key, score):
    """Keep one row per `key`, the one with the highest `score` (e.g. SNR)."""
    order = np.argsort(-np.asarray(t[score], float))   # high score first
    t = t[order]
    _, first = np.unique(np.asarray(t[key]), return_index=True)
    return t[np.sort(first)]


def _norm_specid(col):
    def _one(s):
        if isinstance(s, bytes):
            s = s.decode("ascii", "ignore")
        return str(s).strip()
    return np.array([_one(s) for s in np.asarray(col)])
