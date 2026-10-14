"""Config for the Au18 (Auriga halo 18) GS/E-merger analysis.

Paths, cosmology, snapshot<->time mapping, and GFM_Metals element indices.
All physical units follow auriga_public's load_snapshot transformation
(coords -> physical kpc, masses -> Msun after also multiplying by 1e10,
velocities -> peculiar km/s).  See README note: vector components are [Z,Y,X].
"""
from __future__ import annotations
import numpy as np
from astropy.cosmology import FlatLambdaCDM

# --------------------------------------------------------------------------- #
# Data locations
# --------------------------------------------------------------------------- #
HALO = "halo_18"
HALO_TAG = "halo_18"
SIM_DIR = "/home/hz420/austreams/Auriga_simulation/halo_18/"          # trailing slash
ACCRETED_DIR = "/home/hz420/austreams/Auriga_simulation/lists/accretedstardata/"  # + halo

# Repo-relative outputs
OUT_DIR = "/data/hz420-2/EoS_explore/auriga/out"
FIG_DIR = "/data/hz420-2/EoS_explore/auriga/figures"

# --------------------------------------------------------------------------- #
# Cosmology (from snapshot headers)
# --------------------------------------------------------------------------- #
H0 = 67.77
OMEGA_M = 0.307
OMEGA_L = 0.693
HUBBLE = 0.6777
COSMO = FlatLambdaCDM(H0=H0, Om0=OMEGA_M)
T0_GYR = COSMO.age(0).value            # age of universe at z=0

SNAP_MIN, SNAP_MAX = 50, 127           # snapshots available for halo_18

# Auriga mass unit: snapshot masses are 1e10 Msun/h; loader divides by h,
# so multiply loaded Masses/GFM_InitialMass by 1e10 to get Msun.
MASS_TO_MSUN = 1e10

# --------------------------------------------------------------------------- #
# GFM_Metals element order (standard Arepo/Auriga 9-species vector)
# --------------------------------------------------------------------------- #
ELEMENTS = ["H", "He", "C", "N", "O", "Ne", "Mg", "Si", "Fe"]
IEL = {e: i for i, e in enumerate(ELEMENTS)}

# Atomic weights (for number-ratio abundances)
ATOMIC_WEIGHT = {"H": 1.008, "He": 4.003, "C": 12.011, "N": 14.007,
                 "O": 15.999, "Ne": 20.180, "Mg": 24.305, "Si": 28.085,
                 "Fe": 55.845}

# Solar reference abundances, 12+log(X/H) (Asplund et al. 2009 photospheric)
SOLAR_12 = {"H": 12.0, "He": 10.93, "C": 8.43, "N": 7.83, "O": 8.69,
            "Ne": 7.93, "Mg": 7.60, "Si": 7.51, "Fe": 7.50}


def a_to_age(a):
    """Scale factor -> cosmic time (age of universe) in Gyr."""
    a = np.asarray(a, float)
    z = 1.0 / np.clip(a, 1e-6, None) - 1.0
    return COSMO.age(z).value


def a_to_lookback(a):
    """Scale factor -> lookback time in Gyr."""
    return T0_GYR - a_to_age(a)


def bracket_abundance(metals, el_num, el_den):
    """[el_num/el_den] from a GFM_Metals mass-fraction array (..., 9).

    Uses number ratios and Asplund solar reference.
    """
    metals = np.asarray(metals, float)
    xn = metals[..., IEL[el_num]] / ATOMIC_WEIGHT[el_num]
    xd = metals[..., IEL[el_den]] / ATOMIC_WEIGHT[el_den]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log10(xn / xd)
    solar = (SOLAR_12[el_num] - 12.0) - (SOLAR_12[el_den] - 12.0) \
        if el_den == "H" else (SOLAR_12[el_num] - SOLAR_12[el_den])
    # general solar number-ratio in dex: (12+logX)_num - (12+logX)_den
    solar = SOLAR_12[el_num] - SOLAR_12[el_den]
    return ratio - solar


if __name__ == "__main__":
    # quick self-test of the time mapping
    for a in [0.3, 0.5, 0.7, 1.0]:
        print(f"a={a}  age={a_to_age(a):.2f} Gyr  lookback={a_to_lookback(a):.2f} Gyr")
