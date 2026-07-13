"""Population masks (chemical + kinematic), all driven by config constants.

Works on any Table that carries the standardised columns: feh, mgfe, alfe,
vtan, ecc (see loaders).  Abundance-only masks let us reproduce the paper's
"chemistry-only" selection; the *_orbit masks add the kinematic operationalisation
of Eos / disc.
"""
from __future__ import annotations

import numpy as np
from . import config as C


def _g(t, name):
    return np.asarray(t[name], float)


def mgfe_split(feh):
    """High/low-alpha boundary in [Mg/Fe] as a function of [Fe/H]."""
    return C.MGFE_SPLIT_INT + C.MGFE_SPLIT_SLOPE * np.asarray(feh, float)


# --------------------------------------------------------------------------- #
# Chemistry-only population masks (paper's primary selection)
# --------------------------------------------------------------------------- #
def is_insitu(t):
    return _g(t, "alfe") > C.AL_INSITU


def is_accreted(t):
    return _g(t, "alfe") <= C.AL_INSITU


def is_highalpha(t):
    feh, mgfe = _g(t, "feh"), _g(t, "mgfe")
    return is_insitu(t) & (mgfe > mgfe_split(feh))


def is_lowalpha(t):
    """Low-alpha (thin-disc) chemical sequence, incl. the metal-poor extension.

    Below the high/low-alpha boundary in [Mg/Fe], in-situ ([Al/Fe] high), and
    additionally protected from high-alpha contamination by the diagonal
    [Al/Fe] > 0.9 [Fe/H] + 0.9 line (paper, explicit).
    """
    feh, mgfe, alfe = _g(t, "feh"), _g(t, "mgfe"), _g(t, "alfe")
    below_split = mgfe <= mgfe_split(feh)
    # The diagonal anti-contamination line only cleans the metal-poor *extension*
    # (feh < FEH_EXT_APPLY); the bulk metal-rich thin disc is not subject to it.
    diag = alfe > (C.AL_LOWA_SLOPE * feh + C.AL_LOWA_INT)
    anti_contam = (feh >= C.FEH_EXT_APPLY) | diag
    return is_insitu(t) & below_split & anti_contam & (feh > C.FEH_LOWA_MIN)


# --------------------------------------------------------------------------- #
# Kinematic operationalisations
# --------------------------------------------------------------------------- #
def is_halo_orbit(t):
    vtan, ecc = _g(t, "vtan"), _g(t, "ecc")
    return (np.abs(vtan) < C.EOS_VTAN_MAX) & (ecc > C.EOS_ECC_MIN)


def is_disc_orbit(t):
    vtan, ecc = _g(t, "vtan"), _g(t, "ecc")
    return (vtan > C.DISC_VTAN_MIN) & (ecc < C.DISC_ECC_MAX)


# --------------------------------------------------------------------------- #
# Final science samples
# --------------------------------------------------------------------------- #
def eos_mask(t):
    """Eos: metal-poor low-alpha chemistry on halo-like orbits."""
    feh = _g(t, "feh")
    return (is_lowalpha(t) & is_halo_orbit(t)
            & (feh > C.EOS_FEH_MIN) & (feh < C.EOS_FEH_MAX))


def lowalpha_disc_mask(t):
    """Rotationally-supported low-alpha (thin) disc."""
    return is_lowalpha(t) & is_disc_orbit(t)


def splash_mask(t):
    """Splash: heated high-alpha disc (paper's Fig 6 definition)."""
    feh, vtan = _g(t, "feh"), _g(t, "vtan")
    return is_highalpha(t) & (feh > C.SPLASH_FEH_MIN) & (np.abs(vtan) < C.SPLASH_VTAN_MAX)


def summary_counts(t):
    """Quick diagnostic of how many stars land in each population."""
    return {
        "n_total": len(t),
        "accreted": int(is_accreted(t).sum()),
        "highalpha": int(is_highalpha(t).sum()),
        "lowalpha": int(is_lowalpha(t).sum()),
        "eos": int(eos_mask(t).sum()),
        "lowalpha_disc": int(lowalpha_disc_mask(t).sum()),
        "splash": int(splash_mask(t).sum()),
    }
