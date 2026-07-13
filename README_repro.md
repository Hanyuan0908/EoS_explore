# Eos reproduction — working notes (to continue on another machine)

Investigating the nature of **Eos** (metal-poor in-situ low-α population on halo orbits):
heated low-α disc vs. onset of the low-α disc. Analysis is driven by Belokurov's own
`eos-figures` code applied to our APOGEE + LAMOST data.

## What's here
- `notebooks/01_load_data.ipynb` — APOGEE: reference `eos_figures` cuts/`make_masks`/plots
  applied to **our** APOGEE cache; reproduces the paper figures.
- `notebooks/02_lamost.ipynb` — LAMOST subgiants (Xiang+2024 MSTO ages + DD-Payne
  chemistry) through the **same** selection; chemical planes, [Fe/H]–Vtan, and the
  APOGEE-AstroNN vs LAMOST-MSTO **age comparison**.
- `scripts_repro/` — scripts that build the caches and (re)generate the notebooks:
  - `build_our_cache.py`   -> `data_repro/our_apogee_dr17_lite_ann.fits.gz`
  - `build_lamost_cache.py`-> `data_repro/our_lamost_subgiant_ddpayne.fits.gz`
  - `build_nb.py` / `build_nb2.py` -> regenerate the two notebooks
- `data_repro/` — the matched caches (committed so notebooks run without the raw catalogues).
- `figures_repro/` — output PNGs.

## Setup on a new machine
1. Re-clone the reference code (git-ignored here):
   `git clone git@github.com:vasilybelokurov/eos-figures.git`  (into this repo root).
2. Env: `astro312` (astropy, numpy>=2, scipy, matplotlib). Note: `eos_figures.stats`
   calls the removed `np.trapz`; the notebooks alias `np.trapz = np.trapezoid`.
3. **Absolute paths**: the notebooks/scripts hard-code
   `/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/...` and the raw
   catalogue paths under `/Users/hanyuan/Desktop/...`. If the new machine's paths differ,
   update the `REPO`/`DATA`/catalogue paths at the top of the scripts and notebook setup cells.

## Key settings / knobs
- LAMOST quality cuts (in `build_lamost_cache.py`): `AGE < 14` (drops unphysical MSTO tail),
  Gaia `0.6 < RUWE < 1.4`, chemistry `CHEM_ERR_MAX` on Fe/Mg/Al errors. **Currently 0.15**
  (moderate; tested 0.2/0.15/0.1 — ages robust, sequences sharpen as it tightens).
- Reference selection lines (all from `eos_figures.config.Cuts`): accreted
  `[Mg/Fe]=-0.30[Fe/H]-0.10`, high/low-α `-0.14[Fe/H]+0.135`, in-situ Al `>-0.12`.

## Current result
LAMOST MSTO ages confirm the APOGEE AstroNN story qualitatively: Eos & Splash old, low-α
disc young; LAMOST runs ~2-4 Gyr older (AstroNN saturates above ~10 Gyr). Robust to the
chemistry-error cut. Next: at fixed [Fe/H] in the Eos regime, compare ages of halo-orbit
(Eos) vs disc-orbit low-α stars — the decisive heated-vs-onset test.
