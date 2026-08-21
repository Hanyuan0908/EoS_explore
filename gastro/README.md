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
