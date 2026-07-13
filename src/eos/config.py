"""Central configuration: data paths, unit scalings, and all selection constants.

Selection lines follow the Belokurov & Kravtsov (2022) philosophy and are tuned
visually against the reproduction of Figure 1 of the draft (Non_rotating_low_alpha.pdf).
Everything that defines a population lives here so it is documented and tunable.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Data locations
# --------------------------------------------------------------------------- #
APOGEE_DIR = "/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE"
LAMOST_DIR = "/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/LAMOST"

APOGEE_ASTRONN = os.path.join(APOGEE_DIR, "apogee_astroNN-DR17.fits")
APOGEE_ALLSTAR = os.path.join(APOGEE_DIR, "APOGEE_DR17_all.fits")  # ASPCAP chemistry
APOGEE_ANDERS23 = os.path.join(APOGEE_DIR, "APOGEE_AstroNNdist_Anders23age_BJdist.fits")

LAMOST_SUBGIANT = os.path.join(LAMOST_DIR, "subgiant_fullparam_update.fits")
LAMOST_SUBGIANT_XIANG = os.path.join(LAMOST_DIR, "LAMOST_Gaia_subgiants_Xiangetal2024.fits")
LAMOST_DDPAYNE = os.path.join(LAMOST_DIR, "LMDR9_DDPAYNE_recommend_202505.fits")

# Repo-relative output dirs (filled in by loaders.repo_path)
FIGURE_DIR = "figures"
RESULTS_DIR = "results"

# --------------------------------------------------------------------------- #
# Unit scalings (match the paper's axis notation)
# --------------------------------------------------------------------------- #
E_SCALE = 1e-5     # Energy [km^2/s^2] * E_SCALE  -> paper's "E x 10^-5" (~ -0.4..-0.8)
LZ_SCALE = 1e-3    # Lz [kpc km/s]    * LZ_SCALE -> paper's "Lz x 10^-3" (~ 0..2.5)

# --------------------------------------------------------------------------- #
# Chemical selection lines (B&K22-style; tune against Fig 1)
# --------------------------------------------------------------------------- #
# In-situ vs accreted: in-situ has enhanced aluminium.
AL_INSITU = -0.12          # [Al/Fe] > AL_INSITU  => in-situ  (paper text)

# Anti-contamination of the low-alpha sequence by high-alpha stars (paper, explicit):
#   keep low-alpha only where  [Al/Fe] > AL_LOWA_SLOPE*[Fe/H] + AL_LOWA_INT
AL_LOWA_SLOPE = 0.9
AL_LOWA_INT = 0.9
FEH_EXT_APPLY = -0.6       # diagonal anti-contamination applies only below this [Fe/H]

# High-alpha / low-alpha split in [Mg/Fe]-[Fe/H].
# Modelled as a line in [Mg/Fe] that declines with [Fe/H]:
#   boundary(feh) = MGFE_SPLIT_INT + MGFE_SPLIT_SLOPE * feh
#   high-alpha:  [Mg/Fe] >  boundary
#   low-alpha:   [Mg/Fe] <= boundary
# Starting values from B&K22-style separations; tune against Fig 1 / Fig 4.
MGFE_SPLIT_INT = 0.16
MGFE_SPLIT_SLOPE = -0.08

# Accreted (GS/E) selection used for Figure 2 (kinematic assist):
ACCRETED_LZ_MAX = 500.0    # |Lz| < 500 km/s kpc for the accreted column in Fig 2

# Low-alpha metallicity extension: the regime where Eos lives.
FEH_LOWA_MIN = -1.1        # extend low-alpha disc down to ~ -1
EOS_FEH_MAX = -0.5         # Eos regime: metal-poor end of the low-alpha sequence
EOS_FEH_MIN = -1.1

# --------------------------------------------------------------------------- #
# Kinematic operationalisation of "halo-like orbit" (Eos) vs disc
# (cross-checked against Fig 3 E-Lz clump and Fig 5 Vtan). Tunable.
# --------------------------------------------------------------------------- #
EOS_VTAN_MAX = 100.0       # km/s : low net rotation (Eos sits near Vtan ~ 0)
EOS_ECC_MIN = 0.6          # high eccentricity (paper: 0.8 < e < 1 for the core)
DISC_VTAN_MIN = 150.0      # km/s : rotationally supported low-alpha disc
DISC_ECC_MAX = 0.35        # near-circular

# Splash (heated high-alpha disc) -- paper's Fig 6 definition.
SPLASH_FEH_MIN = -0.9
SPLASH_VTAN_MAX = 80.0

# Quality / sample cuts (ASPCAP allStar cleaning, B&K22-style).
HELIO_DIST_MAX = 15.0      # kpc (paper)
AGE_REL_ERR_MAX = 0.20     # relative age-error cut for age figures (paper)
SNR_MIN = 70.0             # spectral S/N
LOGG_GIANT = (1.0, 3.5)    # red-giant selection in logg
ASPCAP_STARBAD_BIT = 23    # ASPCAPFLAG STAR_BAD bit -> reject if set
# Per-element abundance flags (== 0 means reliable) and EXTRATARG==0 (removes
# commissioning/telluric/duplicate visits) implement the "clean of duplicates and
# unreliable measurements" step of the paper.
