# McMillan17 figure series — observational figures with from-scratch kinematics

Parallel rebuild of the `Fig_paper/obs_*` observational figures, with **all orbital
kinematics derived by us** instead of taken from the AstroNN VAC.

## What changed
- **Distances**: Bailer-Jones et al. 2021 photogeometric (`GAIAEDR3_R_MED_PHOTOGEO`).
- **Astrometry + RV**: Gaia DR3 (`RAdeg/DEdeg/pmRA/pmDE`, `RV`) from the APOGEE DR17 x
  Gaia DR3 TOPCAT crossmatch in `APOGEE_DR17_all.fits`.
- **Orbits/actions**: AGAMA + **McMillan (2017)** potential (was galpy/MWPotential2014).
- Cache: `data_repro/apogee_mcmillan17_kin.fits` (built by
  `scripts_repro/build_apogee_mcmillan17_kin.py`); masks via `scripts_repro/eos_mcm.py`.

## Base cleaning
Identical to the reference except the AstroNN galactocentric velocity errors
(unavailable from scratch) are replaced by **RUWE < 1.4 & fractional BJ21 distance
error < 0.2**, plus a bound-orbit sanity cut (E<0, r_apo<50 kpc). Base = 187,365.

## Eos anchor
Eos **n = 325 (183 metal-poor / 142 metal-rich)**, branch median [Fe/H] = -0.71 / -0.47,
vs the AstroNN anchor 353 (191/162): ~8% shift from the new potential + BJ21 + DR3 RV.
The science (branch age gradient, J_R split, N-dispersion) is unchanged.

## Figures (each writes PDF+PNG to ../Fig_paper_mcmillan17/)
obs_eos_age_dist, obs_energy_pops, obs_mg_al_meanal, obs_alfe_pops, obs_nfe_pops,
obs_cfe_pops, obs_ndispersion, obs_lowa_vtan_pixels, obs_amr_agedist,
obs_eos_branches_overview. (Simulation figures were left untouched, as requested.)
