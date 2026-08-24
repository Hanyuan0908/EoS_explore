# gastro / joaorun003 (Amarante) models

Idealised GASOLINE disc-galaxy runs with an infalling dwarf, used as the second
simulation test of the Eos scenarios alongside `../auriga` (Au18).

## What is here

| file | |
|---|---|
| `jrun003.dwarfM06XY138Z37Vxy20FB20/` | **clumpy + merger** (FB20) = `c.r.c03`. Full snapshot series `.00050`–`.01000` (20 outputs, 0.5 Gyr apart), the `.FeMassFrac`/`.OxMassFrac`/`.iord`/`.timeform` aux arrays for the final snapshot, and the seven `*form.npy` birth arrays (x/y/z, vx/vy/vz, jz). |
| `jrun003.dwarfM06XY138Z37Vxy20.01000` | **not clumpy + merger**, t = 10 Gyr. **Truncated — unusable** (42 MB of an expected 159 MB; the source `.gz` is corrupt: `invalid compressed data--format violated`). Needs re-fetching from the collaborator. |
| `jrun003.param` | Unit definition, **reconstructed here** — see below. |
| `gastro-hanyuan.ipynb` | The collaborator's example notebook (loading, `pid`, accreted flags, plotting). |
| `gastro_config.py` | Paths, unit constants, Eos/disc cuts, the satellite-birth definition, loading + `star_frame`. |
| `prep_gastro.py` | Loads, aligns and solves orbits once per model → `out/<model>_stars.npz`. |
| `ana_gastro_age_kinematics.py` | The age/kinematics figure. Reads the cache, so it is cheap to re-run. |
| `gastro_fig5_prep.py` | One pass over the series for the Borbolato et al. (2026) Fig. 5 reproduction. Caches `v_phi` and `R` for **every** star at **every** snapshot, so cuts stay a plotting-time decision. |
| `ana_gastro_fig5.py` | The Fig. 5 reproduction itself (`figures/gastro_fig5_clumpy_merger.png`). |

Run with the pynbody environment:
`/data/ioasoft/software/miniforge3/envs/python-3.11-2026-01a/bin/python3`
(2.3.0; the `astro312` env used for Auriga has no pynbody).

## Units

The original run directory's `.param` is not available, so pynbody falls back to
defaults and every mass, velocity and time comes out wrong. `jrun003.param`
restores them from the snapshot itself:

* header `time` = 10 code units and this is the t = 10 Gyr output ⇒ the time unit
  is ≈ 1 Gyr;
* `dKpcUnit = 1.0`, `dMsolUnit = 2.325e5` is the standard GASOLINE system in which
  the velocity unit is exactly 1 km/s and the time unit is 0.978 Gyr.

That gives M⋆ = 6.2e10, M_dm = 1.3e12, M_gas = 7.9e10 M⊙ and m⋆ = 2.8e4 M⊙ — all
Milky-Way-like, so the reconstruction is almost certainly the original system.
Positions were already in kpc.

## Units cross-check

Independent of the reasoning above: the z=0 dark-matter mass spectrum splits into
three components, one of exactly 100,000 particles totalling **8.846e10 M⊙**.
Borbolato et al. (2026) §2.2 give the satellite a "total dark matter... mass of
8.83e10 M⊙" — a 0.2 % match, which fixes the unit system independently.

## Remaining limitations

* **`nonClumpy+merger` is unusable** (see above). It is the paper's control: their
  headline claim is that a GSE-like merger *without* clumps fails to make a Splash.
* **The `Isolated Clumpy` model (`c.iso`) is absent.** Figure 5 of Borbolato et al.
  is four columns — Isolated Clumpy and Clumpy+merger — so only the right-hand
  half is reproduced here.
* **No accreted-particle id list** (`{name}_pid_accreted.npy`). Satellite-born
  stars are identified from their birth site instead — see `satellite_born()` in
  `gastro_config.py`. That recovers ~85 % of the dwarf's quoted stellar mass, and
  the residual leaks into the high-α Splash at the 3 % level. The id list is a
  small file and would settle it exactly.
* **Aux abundance arrays exist for the final snapshot only.** Not a problem for
  Fig. 5: populations are selected at z=0 and tracked backwards by array index.
## Selection cuts (all taken from the paper, not inferred)

| cut | value | source |
|---|---|---|
| α tracer | oxygen, [O/Fe] | their footnote 5 |
| low-α | [O/Fe] < −0.13 | printed in their Fig. 3, col. 4 |
| high-α | [O/Fe] > +0.10 | same; the span between is their exclusion "gap" |
| metallicity floor | **[Fe/H] > −1.0** | their §3.1 |
| volume | R_GC > 5 kpc | their §3.2 |
| Splash | V_φ < 100 (low-α), V_φ < 50 (high-α) | their §3.2; no eccentricity cut |
| formation time | t_form < 4 Gyr | printed on their Fig. 3 |

The **[Fe/H] > −1.0 floor matters far more than it looks**. Without it the low-α
sample picks up a metal-poor tail ([Fe/H] ≈ −1.2) that is chemically odd rather
than disc-like: among stars born before t = 1 Gyr these are 1.7 % of their birth
cohort and sit at [O/Fe] = −0.29 while the cohort sits at +0.28. They are already
dynamically hot, and they drag the early Splash track down by ~25 km/s — enough
that the Splash and disc curves no longer meet at t = 1 Gyr as they do in the
published figure. With the floor applied the disc track reads 184 km/s at
t = 1 Gyr against the 182 read off their figure.

The **t_form < 4 Gyr cut touches only the canonical disc** — every Splash star
here is older than that already. It brings the disc track to 241 km/s at
t = 10 Gyr against the 240 read off their figure (264 without it), and cuts the
disc's rms residual from 22.8 to 16.8 km/s.

Measuring the [O/Fe] valley independently (deepest minimum over −0.7 < [Fe/H] <
−0.2) gives −0.155, consistent with their −0.13; the scripts use their value.

## Splash kinematic cut: two versions

`ana_gastro_fig5.py` runs in two modes, writing separate files:

| mode | Splash definition | figure |
|---|---|---|
| `paper` (default) | V_φ < 100 (low-α), V_φ < 50 (high-α) — one-sided, their own | `gastro_fig5_clumpy_merger.png` |
| `symmetric` | −80 < V_φ < +80 for **both** — this project's observational mask (`SPLASH_VTAN_MAX` in `../src/eos/config.py`) | `gastro_fig5_clumpy_merger_vphi80.png` |

The symmetric window drops the retrograde tail a one-sided cut keeps and applies
one threshold to both populations, so low- and high-α are selected like for like.
Sample sizes move accordingly: low-α 487 → 323 (losing 80 < V_φ < 100, nothing
below −80), high-α 4,020 → 6,719 (gaining 50 ≤ V_φ < 80, losing 93 retrograde).

It changes the reading of the figure. Under the paper's cuts the two Splash
tracks sit ~55 km/s apart after t = 4 Gyr, which looks like two dynamically
distinct populations; under one common cut they converge to within **11 km/s**
(mean separation over the whole track 61 → 21 km/s). Most of the apparent
difference was the selection, not the galaxy — which supports their "common
dynamical origin" claim on a like-for-like comparison rather than in spite of one.

## How close the tracks get

Against values read off their published panel, with the full selection above:

| | t=1 | t=1.5 | t=2 | t=2.5 | t=3.5 | t=9.5 |
|---|---|---|---|---|---|---|
| Splash, here | 156 | 142 | 118 | 117 | 73 | 82 |
| Splash, paper | 185 | 150 | 140 | 143 | 85 | 85 |
| disc, here | 184 | 185 | 195 | 210 | 224 | 261 |
| disc, paper | 182 | 190 | 200 | 207 | 211 | 243 |

The **t = 1 Gyr point rests on 30 stars**; bootstrapping its median gives a 95 %
interval of [134, 190], which contains their 185. That point is small-number
noise, not a systematic error.

What remains is a genuine mild systematic from t = 2 onward: the Splash track
runs ~11 km/s slow and the disc ~12 km/s fast, i.e. this measurement separates
the two populations slightly more than theirs does. Median is the best-matching
statistic (rms 14.4 km/s for the Splash, against 16.6 for the mean and 17.4 for a
Gaussian-fit centre), so the residual is not a choice of statistic. The most
likely cause is a small difference in the [O/Fe] calibration, which is drawn in
their Fig. 3 but never quoted numerically.

## Radial cut

Borbolato et al. describe two things that read as if they conflict: normalising
the disc length by each simulation's scale length (§3.2, for Figure 3), and
excluding stars at R_GC < 5 kpc. Taking the **5 kpc literally** is what
reproduces their result — it gives Splash fractions of 0.21 % of the low-α and
8.10 % of the high-α population here, against 0.25 % and 8.16 % in their APOGEE
sample. Rescaling by the measured scale length instead (R_d ≈ 1.25 kpc, so
R > 2.4 kpc) gives 0.43 % and 6.73 %, and roughly six times as many low-α Splash
stars as their Figure 5 shows. So the scripts use `R > 5 kpc`.

## Cross-snapshot identity

GASOLINE appends new star particles to the end of the file, so a star's array
index is a stable id. Verified explicitly: at all 20 snapshots the `tform` array
is exactly a prefix of the final snapshot's.

## Merger epoch

Measured from the snapshot rather than assumed: the fraction of stars that are
retrograde today spikes from a ~2 % baseline to 11–14 % for stars born at
**t = 1.5–2.25 Gyr**, and the eccentric fraction turns back up over the same
interval. That is the window marked in the figures.
