# gastro / joaorun003 (Amarante) models

Idealised GASOLINE disc-galaxy runs with an infalling dwarf, used as the second
simulation test of the Eos scenarios alongside `../auriga` (Au18).

## What is here

| file | |
|---|---|
| `jrun003.dwarfM06XY138Z37Vxy20FB20.01000` | **clumpy + merger** (FB20), t = 10 Gyr. Readable. |
| `jrun003.dwarfM06XY138Z37Vxy20.01000` | **not clumpy + merger**, t = 10 Gyr. **Truncated — unusable** (42 MB of an expected 159 MB; the source `.gz` is corrupt: `invalid compressed data--format violated`). Needs re-fetching from the collaborator. |
| `jrun003.param` | Unit definition, **reconstructed here** — see below. |
| `gastro-hanyuan.ipynb` | The collaborator's example notebook (loading, `pid`, accreted flags, plotting). |
| `gastro_config.py` | Paths, unit constants, Eos/disc cuts, loading + `star_frame`. |
| `prep_gastro.py` | Loads, aligns and solves orbits once per model → `out/<model>_stars.npz`. |
| `ana_gastro_age_kinematics.py` | The age/kinematics figure. Reads the cache, so it is cheap to re-run. |

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

## What these snapshots cannot do

* **No `.FeMassFrac` / `.OxMassFrac`.** Only total `metals`, so there is no
  `[Fe/H]` and no `[O/Fe]`: metallicity is `[M/H] = log10(Z/0.0142)` and there is
  **no α split**. The Eos analogue here is therefore kinematic, not low-α.
* **No accreted-particle id lists** (`*_pid_accreted.npy` live in the
  collaborator's directory), so merger debris cannot be labelled directly.
* **Only the final snapshot**, so there are no birth kinematics — the born-hot vs
  heated test is done on the Auriga side only.

## Merger epoch

Measured from the snapshot rather than assumed: the fraction of stars that are
retrograde today spikes from a ~2 % baseline to 11–14 % for stars born at
**t = 1.5–2.25 Gyr**, and the eccentric fraction turns back up over the same
interval. That is the window marked in the figures.
