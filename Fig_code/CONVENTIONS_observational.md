# Observational (APOGEE / LAMOST) figure conventions — deltas to `CONVENTIONS.md`

`CONVENTIONS.md` was written for the Auriga/GASTRO **simulation** figures on the
cluster. The **observational** paper figures (APOGEE + LAMOST) are made on the
**Mac**, from the `scripts_repro/` analysis. This file records only the
**adaptations**. Everything in `CONVENTIONS.md` still applies unchanged unless it
is overridden below — in particular the style `rcParams` block, "no panel titles /
bold `(a)`,`(b)` tags", **save both PDF and PNG**, **rasterize every data-bearing
artist**, square bins under `aspect='equal'`, blank empty cells, disclose per-panel
normalisation, `viridis` for quantities with no natural centre, and the whole
"Mistakes already made" list. Read that file first; this one second.

## Text must be readable — at least as large as the paper caption

All figure text (axis labels, ticks, legends, in-panel annotations) must be
**comparable to or larger than the paper's caption font**. These figures are
shrunk to column width, so start large: axis labels ~18-19, ticks ~15, titles
~18, legends ~11-13 (bigger than the base `CONVENTIONS.md` block). **Always Read
the rendered PNG before calling a figure done** and check for overlaps and
legibility. If enlarging text causes overlaps, respect any hard axis limit the
user set — do not expand the axis; instead free space by capping the tick labels
(data often sits well below the axis top) so text lands in a blank band, or
shrink the offending element.

---

## Interpreter

Run with the **local** env, not the cluster python:

    /Users/hanyuan/miniforge3/envs/astro312/bin/python

It has `agama` 1.0.156 + numpy/scipy/matplotlib. (`eos_figures` is imported from
the reference repo at `eos-figures/`; add it to `sys.path` or run from a cwd where
it resolves.)

---

## Data provenance

**In-repo, Dropbox-synced (portable — a script using only these runs on either
machine):**

| file | contents |
|---|---|
| `data_repro/our_apogee_dr17_lite_ann.fits.gz` | main matched cache (529k): APOGEE DR17 + AstroNN kinematics/ages/[X/Fe] |
| `data_repro/our_apogee_allspecies.fits.gz` | 19-species wide cache (git-ignored, ~134 MB) |
| `data_repro/our_lamost_subgiant_ddpayne.fits.gz` | LAMOST subgiants, Xiang MSTO ages + DD-Payne Mg/Al |

**On this Mac only, under `~/Desktop/PhD_projects/spectroscopic_catalogues/`
(NOT synced) — scripts that need these are Mac-only, the mirror of how the sim
scripts are cluster-only:**

| file | needed for |
|---|---|
| `APOGEE/apogee_astroNN-DR17.fits` | full 6D phase space (actions, L-vector) |
| `APOGEE/APOGEE_DR17_all.fits` | per-element flags (e.g. `N_FE_FLAG`) |
| `APOGEE/APOGEE_AstroNNdist_Anders23age_BJdist.fits` | Anders 2023 spectroscopic ages |
| `APOGEE/APOGEE_DR17_bingoages.fits` | BINGO (Ciucã 2024) C/N ages |

Load the catalogue and population masks through the reference repo:
`eos_figures.data.load_catalog` + `make_masks(cat, Cuts())`. It returns a FITS
recarray — use `.dtype.names`, not `.colnames`.

---

## Canonical selections — the reproducibility anchor

The observational analogue of the sim's "counts must match exactly" check. **Any
Eos figure must reproduce n = 353 (191 α-rich / 162 α-poor)** — the fastest way to
catch a wrong sample.

```python
c = Cuts(); m = make_masks(cat, c)
ecc  = (rap - rperi) / (rap + rperi)
halo = base & ((ecc > 0.7) | (lz < 0))                 # Davies halo
acc_line = c.slope_acc *feh + c.inter_acc              # -0.30*feh - 0.10  (accreted / in-situ)
hl_line  = c.slope_acc2*feh + c.inter_acc2             # -0.14*feh + 0.135 (high-α / low-α)
divider  = 0.317*feh + 0.353                           # Davies α-rich / α-poor split
lowa   = halo & (feh>-0.9) & (feh<-0.2) & (mg>acc_line) & (mg<hl_line) & (al > c.alfe_cut)
eos_hi = lowa & (mg >  divider)          # α-rich (upper)   n=191
eos_lo = lowa & (mg <= divider)          # α-poor (lower)   n=162
#        Eos total = lowa                                   n=353
disc   = thin_al & (galvt > 150)         # low-α disc reference
splash = thick_al & (galvt < 80)         # Splash reference
```

- `galvt` is V_tan ≈ V_φ. Mean V_φ: Eos +4.6±3.0, α-rich +2.7±2.7, α-poor +6.9±5.8 km/s.
- **Do not** revert to the old clipped `thin_al & Vtan<80 & -0.9<[Fe/H]<-0.5` box —
  it truncated the metal-rich α-poor branch (n=64 instead of 162). See
  `../scripts_repro/plot_eos_selection_zoom.py`.

### Age quality cut (all three age catalogues, applied to Eos AND the disc)

    finite & 0 < age < 20 Gyr  &  sigma_age / age < 0.3

Error/age columns: AstroNN `age_model_error`/`age`; Anders `e_spAgeqrCal`/`spAgeqrCal`;
BINGO `age_total_error`/`age_lowess_correct` (BINGO raw age = 10**`pred_logAge`; use the
calibrated `age_lowess_correct`). This cut thins the metal-poor α-rich branch to
~12–13 in Anders/BINGO, so **the branch age split is only shown for AstroNN**; for
Anders/BINGO show the **combined** Eos vs the matched-metallicity disc.

---

## Population colours — observational analogue of the sim palette

Keep these consistent across the observational figures. Chosen to align with
`CONVENTIONS.md` where the physics maps over (GS/E stays violet; the radially-hot
α-rich branch reuses the born-hot tomato, the colder α-poor reuses born-cold blue).

| population | colour | hex | notes |
|---|---|---|---|
| GS/E / accreted | violet | `#8E24AA` | same as CONVENTIONS GS/E marker |
| **Eos** (whole) | red | `#E8112D` | the headline population |
| Eos **α-rich** (upper) | tomato | `#FF6347` | radially hotter (higher J_R) — born-hot analogue |
| Eos **α-poor** (lower) | blue | `#1F6FB2` | colder / disc-adjacent — born-cold analogue |
| high-α disc | teal | `#00897B` | |
| low-α disc | near-black / `Greys` | `#2B2B2B` | |
| Splash | orange | `#E8712B` | |
| chemical selection lines (accreted, high/low-α) | black | `k` | dashed / dotted |
| Davies α-rich/α-poor divider | green | `#2E7D32` | solid |
| background stellar density | `Greys` | | log-density or column-normalised |

**Colour clash to watch:** Eos is red and CONVENTIONS reserves red `#E8112D` for
kinematic cut *lines*. In observational panels the Eos population itself is the red
object, so draw chemical selection lines in **black/green** (as above) rather than
red, and only use red cut-lines in panels where the Eos population is not itself
red. Disclose any per-figure deviation in the caption, as usual.

Terminology in figures and text: **α-rich / α-poor** (hyphenated), **Eos**,
**Splash**, **GS/E**.

---

## Workflow (same as CONVENTIONS.md)

1. Write `fig_paper_<name>.py`; inline the `rcParams` block.
2. Output **PDF + PNG**, same basename, to `../Fig_paper/`.
3. Save the script in `Fig_code/<figure_name>/` and add a row to
   `Fig_code/README.md` (note it as an **observational** figure so it is not
   confused with the Au18/GASTRO chains).
4. These scripts carry **Mac data paths** by design — the observational mirror of
   the cluster-only sim scripts. That is expected, not a portability bug.
