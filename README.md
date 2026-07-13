# EoS_explore

Investigating the nature of **Eos** (a.k.a. "Aura" in the draft text): the
metal-poor, in-situ **low-α** population on **halo-like orbits** in the Milky Way.
Goal: distinguish whether Eos is (1) a *heated* low-α disc (the low-α Splash) or
(2) the *onset* of low-α star formation, before the disc spun up.

**Current finding:** the onset scenario is favoured — see `results/README.md`.

## Layout
- `src/eos/` — shared package
  - `config.py` — all data paths, unit scalings, and tunable selection constants
  - `loaders.py` — FITS readers → standardised columns (`feh,mgfe,alfe,E5,Lz3,vtan,ecc,rap,rperi,age`)
  - `selections.py` — population masks (accreted / high-α / low-α / Eos / disc / Splash)
  - `plotting.py` — shared density/style helpers
- `scripts/`
  - `fig01..fig06` — reproduce the draft paper's Figures 1–6 (APOGEE; validation)
  - `build_lamost_eos.py` — join DD-Payne abundances to LAMOST subgiant MSTO ages
  - `ana_age_distributions.py`, `ana_amr_fixed_feh.py`, `ana_orbits_vs_splash.py` — the discriminating analysis
- `figures/`, `results/` — outputs

## Environment & running
```bash
PY=/Users/hanyuan/miniforge3/envs/astro312/bin/python   # astropy, scipy, matplotlib
$PY scripts/fig01_overview.py        # etc. (run from repo root)
$PY scripts/build_lamost_eos.py      # writes results/lamost_eos_sample.fits
$PY scripts/ana_amr_fixed_feh.py     # the decisive age-at-fixed-[Fe/H] test
```

## Data (external, not in repo)
APOGEE AstroNN DR17 VAC + LAMOST subgiant ages + LAMOST DR9 DD-Payne abundances,
under `~/Desktop/PhD_projects/spectroscopic_catalogues/{APOGEE,LAMOST}` (see
`src/eos/config.py`). Orbits are pre-computed (MWPotential2014) — no galpy needed.
