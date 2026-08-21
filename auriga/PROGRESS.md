# Au18 / Eos–Auriga project — progress log (AI-continuable handoff)

Last updated: 2026-08-21. This file is written so a fresh AI assistant (or the user)
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
[DONE], (3) split the in-situ stars formed around the merger into
born-hot/born-cold-then-heated channels [DONE, §4], (4) measure the age and
kinematic signature of the Eos analogue and compare it with the observational
selection, in Au18 and in the gastro models [DONE, §4], (5) see §5 for what is
still open.

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
- `channels_au18.py` — the single definition of channels A and B, shared downstream.
- `ana_premerger_splash.py` — builds channel C (`out/premerger_splash.npz`).
- `ana_channels_chemistry.py`, `ana_three_channels_chemistry.py` — [X/Fe] comparisons.
- `ana_birth_height.py`, `ana_birth_radii.py`, `ana_birth_time.py`,
  `ana_channel_radial_gradient.py` — the birth-property diagnostics behind the cuts.
- `diag_disc_ABC_montage.py` — face-on/edge-on montages of A, B and C.
- `ana_z0_kinematic_catalog.py` — **the z=0 catalogue** (in-situ + GS/E; kinematics,
  orbits, chemistry). Everything below reads it, so it only needs running once.
- `ana_eos_age_kinematics.py` — Eos analogue selected as in the data; age + orbits.
- `ana_channels_born_hot.py` — born hot vs heated, and what the observational cut
  recovers from each channel.
- `../orbit_tools.py` — spherical-potential r_apo/ecc, density contours, and the
  local-enhancement statistic. Shared with `../gastro`.

Outputs:
- `out/gse_clean_ids.npy` — **the clean GS/E star IDs (use this going forward)**.
- `out/gse_clean_z0.npz` — clean GS/E z=0 birth age, [Fe/H], initial mass.
- `out/matched_z0.npz` — all accreted + all in-situ matched to z=0 (ages, [Fe/H], r, mass).
- `out/gse_track_clean_55_127_2.npz` (+ proxy tracks) — orbital tracks.
- `out/exsitu_assembled.npz`, `out/gse_proxy_ids.npy` — earlier/superseded.
- `out/z0_insitu_catalog.npz` — **1.98M in-situ stars + the GS/E debris** with
  v_phi/v_R/v_z, L_z, E, eps, r_apo, r_peri, ecc, age, [Fe/H] and six [X/Fe].
- `out/eos_two_channels.npz`, `out/merger_birth_radii.npz`, `out/premerger_splash.npz`
  — birth + z=0 properties for the A/B and C parent samples.
- `figures/au18_gse_merger_dating.png`, `au18_gse_merger_montage.png`,
  `au18_gse_anisotropy.png`.
- `figures/au18_eos_age_kinematics.png`, `au18_channels_born_hot.png`,
  `au18_eos_channels_chemistry_clean.png` — the current headline figures.

---

## 4. Sessions 2-3: the Eos channels, and the age/kinematic signature

### Session 2 - three in-situ channels around the merger
In-situ stars were split by *birth* kinematics and birth height, all sharing one
circularity scale (`channels_au18.py`, `ana_premerger_splash.py`):

| channel | definition | N |
|---|---|---|
| **A** heated disc | born cold in the plane during the merger (eps_b>0.7, \|z_b\|<1 kpc), eps_0<0.3 | 948 |
| **B** born radial | born hot off-plane during the merger (eps_b<0.3, \|z_b\|>3 kpc), eps_0<0.3 | 2,642 |
| **C** pre-merger Splash | same as A but formed *before* the merger | 3,831 |

Chemistry result (`ana_channels_chemistry.py`, `au18_eos_channels_chemistry_clean.png`):
GS/E debris sits on its own locus, while **A, B and C overlap almost completely**
in every [X/Fe]; B departs from the in-situ populations only in [Fe/H]. So B
formed from *diluted host gas*, not from GS/E gas.

### Session 3 - kinematic + age signature (the current focus)
A z=0 catalogue of **every** in-situ star was built so the simulation can be cut
the way the *observations* are, instead of by birth properties
(`ana_z0_kinematic_catalog.py` -> `out/z0_insitu_catalog.npz`, 1.98M stars):
v_phi, v_R, v_z, L_z, E, eps, **r_apo, r_peri, eccentricity**, age, [Fe/H] and the
six [X/Fe]; the GS/E debris is measured identically and stored alongside.
Turning points come from the spherically-averaged snapshot potential
(`../orbit_tools.py`, shared with the gastro analysis).

**Applying the APOGEE/LAMOST cuts** (|v_phi|<100 km/s, ecc>0.6, 4<R<30 kpc) to the
in-situ stars gives 47,270 "Eos-like" stars, median age 10.35 Gyr, [Fe/H]=-0.54
(`ana_eos_age_kinematics.py`, `au18_eos_age_kinematics.png`):

* the Eos analogue is **old** - median 10.4 Gyr vs 4.8 Gyr for disc orbits, and it
  sits between the disc and the GS/E debris (11.4 Gyr) in age and in r_apo
  (14.1 vs 10.2 and 25.5 kpc);
* star formation into this channel is **enhanced x2.9 at the merger**: 14% of the
  cohort born at t=4.8-5.6 Gyr ends up Eos-like, against 5% in the intervals
  either side. It is a burst on the plunge, not a long-lived channel;
* **age at fixed [Fe/H]**: over the observed Eos regime (-1.1<[Fe/H]<-0.5) the two
  tracks agree (-0.14 Gyr) - Au18 does *not* reproduce the observed offset there.
  The +2 to +3 Gyr offset appears only at [Fe/H]>-0.25, where "hot orbit" mostly
  means old bulge. Note Au18's in-situ [Fe/H] runs ~0.4 dex high, so the observed
  window maps onto the metal-poor tail of the simulation.

**Born hot or heated** (`ana_channels_born_hot.py`, `au18_channels_born_hot.png`) -
the test the data cannot do:

| channel | median d(eps) since birth | median r_apo | % passing the observational Eos cut |
|---|---|---|---|
| A heated disc | **-0.73** (heated) | 4.0 kpc | 81% |
| B born radial | **-0.08** (born hot) | **16.0 kpc** | 59% |
| C pre-merger Splash | **-0.72** (heated) | 5.1 kpc | 47% |

B is the only channel that reaches GS/E-like apocentres; A and C are confined
inside ~5 kpc. So an observer's Eos sample in Au18 would be a mixture, but the
part of it on genuinely Eos-like *orbits* is the born-hot, merger-induced channel.

### gastro / joaorun003 (see ../gastro/README.md)
Second, idealised test with an infalling dwarf. Only the **clumpy+merger** model
is usable (the not-clumpy snapshot is truncated - corrupt .gz - and needs
re-fetching), there is no alpha information and only the final snapshot, so this
side contributes the age/orbit comparison only. The merger is located from the
snapshot itself at **t=1.5-2.25 Gyr** (retrograde fraction spikes ~7x). Result
(`gastro_eos_age_kinematics.png`): Eos-like stars have median age 9.3 Gyr vs
4.3 Gyr for disc orbits, are metal-poorer, and their formation shows **two
peaks** - an early pre-spin-up one and a distinct merger-induced burst
(x1.7 over the flanking intervals; 27% of all Eos-like stars are born in the
merger window).

---

## 5. NEXT

Open threads, in rough priority order:

1. **Re-fetch the not-clumpy gastro snapshot** (and, if possible, the
   `.FeMassFrac`/`.OxMassFrac` aux arrays, the accreted-pid lists and more than
   one snapshot). Clumpy vs not-clumpy is the comparison the gastro side exists
   to make, and right now only one half of it is readable.
2. **The [Fe/H] scale mismatch** in Au18. Either match on metallicity *rank*
   rather than absolute [Fe/H], or restrict to the simulation's own metal-poor
   tail, before concluding anything from the age-at-fixed-[Fe/H] panel.
3. **Radial dependence** of the Eos-like fraction: everything above is 4<R<30 kpc,
   whereas the data are solar-neighbourhood.

Earlier open decisions, still unresolved: (a) GS/E-debris purity vs an E-Lz cut;
(b) whether the in-situ birth-time window should reach earlier than 4 Gyr.
