# Au18 / Eos–Auriga project — progress log (AI-continuable handoff)

Last updated: 2026-07-21. This file is written so a fresh AI assistant (or the user)
can pick up the work with full context. Read this first, then `auriga/README.md`.

---

## 0. Big-picture goal

Parent project `EoS_explore` asks whether **Eos** (metal-poor, in-situ **low-α**
stars on **halo-like/non-rotating orbits**; called "Aura" in the draft
`../Non_rotating_low_alpha.pdf`) is:
- **(H) heated** low-α disc (a low-α analogue of the Splash), or
- **(O) onset** of low-α star formation — born kinematically hot *before* the disc
  spun up ("upside-down" formation).

Observations (APOGEE+LAMOST, in `../src`, `../scripts`, `../results`) currently
favour **onset**. We are now using the **Auriga halo 18 (Au18)** cosmological
zoom — a known GS/E-like merger host (Fattahi et al. 2019, stz159) — to test the
scenarios directly, where we have birth times + birth kinematics + provenance.

**Plan:** (1) date the GS/E merger [DONE], (2) verify the GS/E identification
[DONE], (3) select in-situ stars formed around the merger and split into
low-α/high-α × born-hot/born-cold-then-heated to discriminate H vs O [NEXT].

---

## 1. Environment & data (all verified working)

- **Python:** `/data/hz420-2/astro312/bin/python` (has numpy/scipy/astropy/h5py +
  `auriga_public`). Use this for everything.
- **Analysis package:** `auriga_public` (bitbucket grandrt/auriga_public).
  A copy is at `auriga/auriga_public_src`. NOTE: the *imported* module actually
  resolves to astro312's site-packages copy, which is a slightly different version
  than the source copy (see gotchas).
- **Simulation:** `/home/hz420/austreams/Auriga_simulation/halo_18/`
  - `snapdir_050..127` (snapshots; z≈3.7→0) and `groups_050..127` (Subfind).
    Snapshots below 50 are NOT present. Level-4 res (DM 3.4e5, star ~5e4 Msun).
  - **No `trees_sf1` merger tree for halo_18** (only halos 26,27 have it).
- **Accreted-star provenance:**
  `/home/hz420/austreams/Auriga_simulation/lists/accretedstardata/halo_18/`
  `halo_18starID_accreted_all_newmtree_NNN.hdf5`. Read with
  `auriga_public.util.read_starparticle_mergertree_data_hdf5(127, ACCRETED_DIR, "halo_18")`
  → assembles 1.48M ex-situ + 1.98M in-situ unique star IDs, each with `BirthSnap`.

### Units & conventions (IMPORTANT)
- `load_snapshot(..., applytransformationfacs=True)` gives **physical** units:
  coordinates in **Mpc** (×a/h), masses /h (multiply loaded `Masses`/`GFM_InitialMass`
  by **1e10** for Msun), velocities peculiar km/s (×√a).
- **Wind particles** have `GFM_StellarFormationTime < 0` → keep only `> 0` for stars.
- `GFM_Metals` = 9 species `[H,He,C,N,O,Ne,Mg,Si,Fe]` — **no Al** (so no [Al/Fe]
  in-situ/accreted split; use provenance labels + kinematics instead).
- Raw snapshot vector components are ordered `[Z,Y,X]`.
- `GFM_StellarFormationTime` = birth **scale factor** → cosmic time via
  `config_au18.a_to_age`.

### Gotchas already hit (don't repeat)
- The single per-snapshot provenance file (e.g. `_127`) holds only that snapshot's
  *increment* (all young). Must use the assembled reader for the full catalog.
- Provenance `RootIndex`/`PeakMassIndex`/`PeakMassInfalltime` are **per-snapshot
  tree indices**, NOT consistent global progenitor labels — do NOT group by them.
- The z=0 accreted catalog is dominated by **surviving satellites at 300–400 kpc**
  (young, metal-rich). GS/E is the *disrupted* progenitor in the inner halo.
- `util.align_galaxy(s, radialcut=0.01)` **rotates in place, returns None**, and
  puts the disc angular momentum on **component 0** → disc plane = components (1,2),
  rotation/symmetry axis = component 0. (Validated: young disc ⟨vφ⟩=255, σ_z=26,
  scale-height 0.3 kpc.)
- Installed `util` lacks `calculate_bulk_velocity` — compute bulk velocity inline
  (mass-weighted mean within 10 kpc).

---

## 2. What's been done + key results

### GS/E identification (clean)
The GS/E = **the single most massive satellite**: Subfind subhalo **537 at snap 62**
(M⋆=1.04e9). Selected its members by a phase-space sphere (|Δx|<25 kpc, |Δv|<200 km/s)
→ `out/gse_clean_ids.npy` (**21,487 stars, M⋆=1.56e9 Msun, [Fe/H]=−0.82**). Traced to
z=0: median r=22 kpc, 85% within 50 kpc → fully phase-mixed (disrupted, not a survivor).
(An earlier inner-halo "proxy", `out/gse_proxy_ids.npy`, was contaminated — superseded.)

### Merger dating — three clocks agree
| clock | value |
|---|---|
| first apocentre (bound core, MAD~1.5 kpc) | t≈3.0–3.5 Gyr, r≈215 kpc |
| pericentre plunge | t≈5.0 Gyr (z≈1.25) |
| **coalescence / phase-mix (MAD inflates to ~10 kpc)** | **t≈5.3–5.6 Gyr, z≈1.1–1.2, lookback≈8.4 Gyr** |
| in-situ merger-induced SFR burst peak | t≈5.25 Gyr (z≈1.17) |
| GS/E own SF truncation (90 pct birth) | t≈3.2 Gyr |

### z=0 kinematics (disc-aligned) — the GS/E "sausage"
GS/E debris ⟨vφ⟩≈−5 km/s (non-rotating), σ_R=206/σ_φ=111/σ_z=135 km/s,
**spherical β≈0.90–0.95** (0.92 over r=5–40 kpc). In-situ disc rotates at +230 km/s.

### Consistency with Fattahi et al. 2019 (stz159)
Au18 is **named an *extreme* GS/E analogue** (Au-5,9,10,18). Their GS/E analogues:
merger 4–8 Gyr (6–10 Gyr ago), M⋆ 1e9–1e10, β>0.8, [Fe/H]~−1. Our Au18 values
(t≈5.3 Gyr / lookback 8.4 Gyr, M⋆=1.6e9, β_sph≈0.9, [Fe/H]≈−0.8) all match.

### Classification subtlety (central to the science)
Provenance "ex-situ" is by **birth location**. Stars formed from GS/E *gas* inside
the main galaxy are labelled **in-situ**, not accreted, and are exactly the
merger-triggered Eos/Splash analogues. So the Eos candidates must be selected from
the **snapshots** by birth time + chemistry + kinematics, NOT from the accreted list.

---

## 3. Files (in `auriga/`)

Scripts (run with the astro312 python, from `auriga/`):
- `config_au18.py` — paths, cosmology, snap↔time, `GFM_Metals` indices,
  `a_to_age`, `bracket_abundance([Fe/H],[Mg/Fe])`.
- `date_merger_z0.py` — early z=0 exploration (uses provenance; largely superseded
  by the clean-satellite method but kept for the in-situ/accreted matching).
- `date_merger_track.py <s0> <s1> <step> [ids_file] [tag]` — orbital track of a
  star-ID set across snapshots (median galactocentric r + clump dispersion).
- `diag_merger_montage.py` — 3×4 spatial montage of the merger (uses `gse_clean_ids`).
- `diag_anisotropy.py` — disc-aligned vR–vφ + β (cyl & sph) for clean GS/E vs disc.
- `figure_date_merger.py` — 3-panel dating summary (orbit / SFH / chemistry).

Outputs:
- `out/gse_clean_ids.npy` — **the clean GS/E star IDs (use this going forward)**.
- `out/gse_clean_z0.npz` — clean GS/E z=0 birth age, [Fe/H], initial mass.
- `out/matched_z0.npz` — all accreted + all in-situ matched to z=0 (ages, [Fe/H], r, mass).
- `out/gse_track_clean_55_127_2.npz` (+ proxy tracks) — orbital tracks.
- `out/exsitu_assembled.npz`, `out/gse_proxy_ids.npy` — earlier/superseded.
- `figures/au18_gse_merger_dating.png`, `au18_gse_merger_montage.png`,
  `au18_gse_anisotropy.png`.

---

## 4. NEXT (session 2, awaiting user go-ahead)

Select **in-situ** stars (all z=0 main-galaxy stars minus ex-situ) born in the
merger window and split to discriminate Eos scenarios H vs O:
1. Reuse `out/gse_clean_ids.npy` and the validated disc-aligned frame in
   `diag_anisotropy.py`.
2. In-situ merger-epoch sample: birth time ~t=4–6 Gyr (centred on coalescence 5.3;
   possibly extend earlier to catch the pre-merger onset phase — decision pending).
3. Chemistry: [Mg/Fe]–[Fe/H] low-α vs high-α (Mg=idx6, Fe=idx8).
4. Kinematics: compare **birth** vs **present-day** vφ/eccentricity — "born hot"
   (onset) vs "born cold then heated" (heated) — the direct H-vs-O test.
5. Identify the Eos analogue = in-situ low-α on hot/non-rotating orbits; check
   whether it is born hot (favours O) or heated after forming cold (favours H).

Open decisions for the user: (a) GS/E-debris purity vs an E–Lz cut; (b) exact
in-situ birth-time window (centred on 5.3 Gyr vs reaching earlier).
