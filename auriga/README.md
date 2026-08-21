# Au18 (Auriga halo 18) — dating the GS/E-analogue merger

Goal: use Au18 (a GS/E-like merger host, Fattahi et al. 2019) to test the two
scenarios for **Eos** — *heated low-α disc* vs *onset of low-α star formation
before disc spin-up* — by comparing the chemodynamics of stars formed around and
before the GS/E merger. **Session 1 (this):** date the GS/E merger.

## Data & tooling
- Sim: `/home/hz420/austreams/Auriga_simulation/halo_18/` — snaps 50→127
  (z≈3.7→0), Subfind groups, level-4 (DM 3.4e5, star ~5e4 Msun). No `trees_sf1`
  merger tree for halo_18 (only halos 26,27 have it).
- Accreted-star provenance: `.../lists/accretedstardata/halo_18/…_newmtree_NNN.hdf5`.
  Read with `auriga_public.util.read_starparticle_mergertree_data_hdf5` (assembles
  all snaps → 1.48M ex-situ + 1.98M in-situ unique stars, each with `BirthSnap`).
  **Caveat:** `RootIndex`/`PeakMassIndex` are *per-snapshot* tree indices, NOT a
  consistent global progenitor label — do not group by them across snapshots.
- Package `auriga_public` installed (editable) at `auriga/auriga_public_src`.
  Python: `/data/hz420-2/astro312/bin/python`.
- Units: `load_snapshot` → coords physical **Mpc**, mass /h (×1e10 for Msun),
  vels peculiar km/s; wind particles have `GFM_StellarFormationTime<0`; vectors
  are ordered `[Z,Y,X]`. `GFM_Metals` = `[H,He,C,N,O,Ne,Mg,Si,Fe]` (no Al).

## Method (scripts here, run with the astro312 python)
1. `date_merger_z0.py` / matching — assemble accreted+in-situ catalogs, match to
   z=0 snapshot for birth time (`GFM_StellarFormationTime`→cosmic time), [Fe/H],
   [Mg/Fe], galactocentric radius. GS/E debris = inner-halo (r<50 kpc) accreted
   stars; the outer accreted stars (r≈300–400 kpc) are surviving satellites.
2. `date_merger_track.py <s0> <s1> <step>` — follow the GS/E-debris star IDs
   (r<60, [Fe/H]<−0.7, born<5 Gyr) across snapshots; median galactocentric
   distance + clump dispersion → orbital decay → coalescence.
3. `figure_date_merger.py` → `figures/au18_gse_merger_dating.png`.

## Clean GS/E identification (supersedes the inner-halo proxy)
The inner-halo "accreted" proxy was contaminated by several structures. The
**single most massive satellite** — subhalo 537 at snap 62 (Subfind
M⋆=1.04e9), selected by a phase-space sphere → `out/gse_clean_ids.npy`
(21,487 stars, M⋆=1.56e9, [Fe/H]=−0.82) — is the GS/E. Traced to z=0 it is fully
phase-mixed (median r=22 kpc, 85% within 50 kpc): a disrupted merger, not a
survivor. Diagnostics:
- `diag_merger_montage.py` → `figures/au18_gse_merger_montage.png` (infall→streams→phase-mix).
- `diag_anisotropy.py` → `figures/au18_gse_anisotropy.png` (vR–vφ sausage + β).

## Result — three independent clocks agree (clean sample)
| clock | value |
|---|---|
| first apocentre (bound core, MAD~1.5 kpc) | t ≈ 3.0–3.5 Gyr at r≈215 kpc |
| pericentre plunge | t ≈ 5.0 Gyr (z≈1.25) |
| **coalescence / phase-mix (MAD inflates)** | **t ≈ 5.3–5.6 Gyr (z≈1.1–1.2, lookback ≈8.4 Gyr)** |
| in-situ merger-induced SFR burst peak | t ≈ 5.25 Gyr (z≈1.17) |
| GS/E own SF truncation (90 pct birth) | t ≈ 3.2 Gyr |

**Kinematics (z=0, disc-aligned):** GS/E debris ⟨vφ⟩≈−5 km/s (non-rotating),
σ_R=206/σ_φ=111/σ_z=135 km/s, **spherical β≈0.90–0.95** (0.92 over r=5–40 kpc) —
the classic radially-anisotropic "sausage".

## Comparison with Fattahi et al. 2019 (stz159) — consistent
| quantity | Fattahi+2019 (GS/E analogues) | this work (Au18) |
|---|---|---|
| Au18 status | **named an *extreme* GS/E analogue** (Au-5,9,10,18) | ✓ |
| merger time | ~4–8 Gyr (6–10 Gyr ago) | t≈5.3 Gyr (lookback 8.4 Gyr) |
| progenitor M⋆ | 10⁹–10¹⁰ M⊙ | 1.6×10⁹ M⊙ |
| anisotropy β | >0.8 for the extreme cases | β_sph≈0.9 |
| metallicity | [Fe/H]~−1 | [Fe/H]≈−0.8 |

The in-situ starburst coincides with coalescence — these merger-epoch **in-situ**
stars are the Eos/Splash analogues, and (per the classification note) they are
labelled **in-situ**, not accreted, so they must be selected from the snapshots by
birth time + chemistry + kinematics, not from the accreted-star list.

## Next (session 2, pending review)
Select in-situ stars born in the merger window (~t=4–6 Gyr) and split into
low-α vs high-α + birth-hot vs born-cold-then-heated to discriminate onset vs
heated for Eos.
