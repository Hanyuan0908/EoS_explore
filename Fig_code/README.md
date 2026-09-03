# Code behind the paper figures

**Read `CONVENTIONS.md` first** -- style, colours, data provenance, and the list
of mistakes already made and fixed.  It is written so a fresh session can match
the existing figures without re-deriving any of it.

A frozen copy of the scripts that produced each figure in `../Fig_paper/`, taken
so the figures stay reproducible even as the working scripts in `../auriga/` and
`../gastro/` keep moving. These are copies, not the live versions: edit the
originals and re-copy rather than editing here.

Run everything with `/data/hz420-2/astro312/bin/python`. Paths inside the scripts
point at the real `out/` directories in `../auriga/` and `../gastro/`, so they run
in place without needing the data duplicated.

---

## `au18_birth_orbits/` -> `Fig_paper/au18_birth_orbits.pdf`

The orbits Au18 stars are born on, through the GS/E merger. Four panels: birth
circularity against cosmic time, its distribution before/during/after, the
star-formation history split by birth orbit, and the halo-born/disc-born ratio.

Classification:

    disc-born   eps > 0.8  OR  z_max < 1.5 kpc
    halo-born   eps <= 0.8 AND z_max >= 1.5 kpc

where `eps = L_z/L_circ(E)` and `z_max` are measured in the first stored snapshot
at or after each star formed, in that epoch's own potential and disc frame.

**Read `METHOD_zmax_from_Jz.md` before touching the z_max part.** It records the
(2/pi) normalisation, the measured accuracy against orbit integration, and the
axis-permutation sign trap that silently negates L_z.

Chain, in order:

| script | writes | ~time |
|---|---|---|
| `ana_z0_kinematic_catalog.py` | `out/z0_insitu_catalog.npz` | — |
| `prep_birth_orbits.py` | `out/snapshot_times.npz` (also the superseded envelope-eps file) | ~10 min |
| `prep_potentials_ref.py` | `out/potentials_ref/*.ini`, 36 AGAMA CylSpline potentials | ~25 min |
| `prep_birth_actions.py` | `out/birth_orbits_actions.npz` — eps, J_r, J_z, J_phi | ~15 min |
| `prep_zmax.py` | `out/birth_orbits_zmax.npz` — z_max from J_z | ~1 min |
| `ana_birth_orbit_sfh.py` | `out/insitu_imass.npz` (cached GFM_InitialMass, in its first block) | ~1 min |
| `fig_paper_birth_orbits.py` | **the figure**, PDF + PNG | seconds |

`diag_jz_to_zmax.py` is the validation: it integrates orbits with `agama.orbit`
and compares the action-derived z_max against the true one. Not needed to make
the figure, kept because it is what justifies the approximation.

Two things that are easy to get wrong and are commented in the code:

- The KDE bandwidths are Gaussian **sigma**, not bin widths. `BW_T = 0.05` Gyr
  reproduces a 0.15 Gyr histogram; `sigma = 0.15` oversmooths the narrow
  halo-born spike and drags the panel-(d) peak from 0.54 down to 0.35.
- `prep_potentials_ref.py` follows `~/python_script/compute_auriga_potential.py`:
  particle types 4, 1, 0 inside 0.5 R200, `Rmax = 50`, `zmax = 20`. An earlier
  version used the low-res DM types and a 400 kpc grid, which gave a
  non-monotonic inner rotation curve and an AGAMA ActionFinder that refused to
  initialise.

Quote the epoch-averaged fractions (6.5 / 22.8 / 5.2 per cent halo-born) from the
script's stdout rather than reading peak values off the smoothed curves.

---

## `au18_birth_positions/` -> `Fig_paper/au18_birth_positions.pdf`

Edge-on view of the same two birth classes, at the GS/E pericentre (t = 4.99 Gyr)
and at a quiescent late epoch (t = 9.41 Gyr).  Shows that the halo-born class is
genuinely off-plane at the merger, and that it is not the bar -- a bar would be a
thin central line in this projection.

`fig_paper_birth_positions.py` reads the same `birth_orbits_actions.npz` and
`birth_orbits_zmax.npz` as the figure above, so the prep chain is identical; the
prep scripts here are symlinks into `au18_birth_orbits/`.

Each panel is normalised to its own peak because the four populations differ in
number by more than twenty times, so the colour shows shape, not abundance.  The
frame is +-25 kpc and the merger halo-born population extends past it; the
fraction outside is annotated on the panel rather than left implicit.

---

## `au18_vr_vphi_three/` -> `Fig_paper/au18_vr_vphi_three.pdf`

The Eos selection in v_R-v_phi: all merger-born stars at z = 0, the stars passing
the Eos cut, and those same stars at birth with the v_phi = 150 km/s split that
separates born-hot from born-cold.

Uses the ORIGINAL merger window, t_form = 4.99-6.54 Gyr, from
`out/merger_birth_vs_z0_kinematics.npz` (built by
`auriga/ana_merger_birth_vs_z0_kinematics.py`) plus `out/z0_insitu_catalog.npz`
for the eccentricity.  Reproduces the published counts exactly: 171,826
merger-born, 7,583 Eos-like, 4,283 born hot, 3,300 born cold.

`ana_merger_vr_vphi_three.py` is the working version with titles and per-panel
statistics; `fig_paper_vr_vphi_three.py` is the paper cut.

---

## `splash_vphi_evolution/` -> `Fig_paper/splash_vphi_evolution.pdf`

V_phi evolution of the low- and high-alpha Splash in the GASTRO clumpy+merger run.

| script | writes |
|---|---|
| `gastro_fig5_prep.py` | `out/fig5_clumpy_merger.npz` |
| `fig_paper_splash_vphi.py` | **the figure**, PDF + PNG |

`gastro_config.py` carries the paths and the shared configuration.

---

## `splash_vphi_evolution_3panel/` -> `Fig_paper/splash_vphi_evolution_3panel.pdf`

The three-panel version of the figure above: the same [O/Fe]-[Fe/H] selection
plane and the same V_phi(t) tracks, with the z=0 V_phi distributions of the two
alpha populations inserted between them so the Splash cut that turns one panel
into the other is visible.  The right-hand panel also carries the observed Eos
rotation, V_phi = +4.6 km/s, as a red reference line -- an external number, not
measured from this run, so the caption must say where it comes from.

The shading around each track is the **uncertainty on the median** -- the 16-84
range of 500 bootstrap medians -- not the 16-84 spread of the stars, which is what
the two-panel version shades.  Say which of the two the shading is in the caption:
the same ribbon means very different things in the two figures.

The true error is 0.7, 1.2, 1.5 and 5.2 km/s for the four tracks (the script
prints them), which is invisible on a 475 km/s axis, so every band in the panel is
drawn **inflated by `ERR_SCALE = 5`** about its central value -- the four tracks
and the +-3 km/s band on the observed Eos line alike, so the two are read on the
same scale.  They are magnified error bars, not intervals anything falls in, and
**nothing on the figure says so: the caption must state the factor.**  Set
`ERR_SCALE = 1` for true widths.

Same chain as `splash_vphi_evolution/`; `gastro_config.py` and
`gastro_fig5_prep.py` here are symlinks into that directory.

| script | writes |
|---|---|
| `gastro_fig5_prep.py` | `out/fig5_clumpy_merger.npz` |
| `fig_paper_splash_vphi3.py` | **the figure**, PDF + PNG |

Run it with the **pynbody** environment, not `astro312` --
`/data/ioasoft/software/miniforge3/envs/python-3.11-2026-01a/bin/python3` --
because `gastro_config.py` imports pynbody.  `paper` as an argument swaps the
symmetric |V_phi| < 80 km/s window for Borbolato et al.'s asymmetric cuts and
writes the `_papercuts` variant.

The two-panel `splash_vphi_evolution` figure is kept as it is; this is an
addition, not a replacement.

---

# Observational figures (APOGEE) — made on the Mac; see `CONVENTIONS_observational.md`

## `obs_mg_al_meanal/` -> `Fig_paper/obs_mg_al_meanal.pdf`

The chemical planes behind the in-situ selection, three panels each with its own
colourbar. (a) [Mg/Fe]-[Fe/H] log-density with the accreted (dashed) and
high-a/low-a (dotted) lines and the accreted / high-a / low-a labels; (b)
[Al/Fe]-[Fe/H] log-density with the in-situ Al cut ([Al/Fe]=-0.12); (c) the
[Mg/Fe]-[Fe/H] plane coloured by mean [Al/Fe], showing Al rises across the
high-a/low-a line. Panels (a),(b) reproduce the Mg/Al panels of
`figures_repro/01_fig1_energy_mg_al.png`; (c) reproduces the mean-[Al/Fe] panel of
`figures_repro/01_fig4_alfe_3pops.png`. Portable: reads only
`data_repro/our_apogee_dr17_lite_ann.fits.gz`. Run with the local `astro312`.

## `obs_energy_pops/` -> `Fig_paper/obs_energy_pops.pdf`

E-Lz for the three populations (accreted / high-a / low-a), log-density in the
(Lz, E) plane with the Lz=0 line; the Eos overdensity is labelled in the low-a
panel. Exact reproduction of `eos_figures.figures.plot_energy_pops`
(== `figures_repro/01_fig3_energy_pops.png`). Portable: reads only
`data_repro/our_apogee_dr17_lite_ann.fits.gz`. Run with the local `astro312`.

## `obs_alfe_pops/` -> `Fig_paper/obs_alfe_pops.pdf`

[Al/Fe]-[Fe/H] for accreted / high-a / low-a. Top row: column-normalised density
with the accreted diagonal and the in-situ Al cut. Bottom row: the same planes
coloured by median V_phi (colourbar on the high-a panel), with the GS/E, Aurora,
Splash+high-a disk, Eos and low-a disk features labelled. Reproduces
`eos_figures.figures.plot_alfe_pops`; only change is the colourbar label
V_tan -> V_phi. Portable. Run with the local `astro312`.

## `obs_lowa_vtan_pixels/` -> `Fig_paper/obs_lowa_vtan_pixels.pdf`

The low-a (in-situ) population in the [Fe/H]-V_phi plane, three pixel panels
(bins=70x70, min_count=1): (a) number counts (cmasher amber, log); (b) mean
apocentric radius r_apo; (c) mean pericentric radius r_peri (both RdYlBu_r).
Colourbars are inset in the freed bottom space (y extended to -250). Shows the
slow/non-rotating Eos foot below ~100 km/s is on plunging eccentric orbits.
Portable. Run with the local `astro312`.

## `obs_nfe_pops/` -> `Fig_paper/obs_nfe_pops.pdf` (+ `obs_cfe_pops.pdf`)

[N/Fe]-[Fe/H] for the two in-situ populations (high-a / low-a), 2x2. Top row:
column-normalised [N/Fe]-[Fe/H] density (Greys) with the P5 and P95 tracks of
[N/Fe] vs [Fe/H] (both red); bottom row: the same planes coloured by median
V_phi. Colourbars at the far right, one per row; equal physical aspect (1 dex
[Fe/H] == 1 dex [N/Fe]). Derived from `scripts_repro/plot_fig2_nfe_pops.py` with
the accreted column, legend, median line and "Eos?" annotation removed.
Element-parametrised (`n_fe` default, `c_fe` optional). Portable. Run with the
local `astro312`.

## `obs_ndispersion/` -> `Fig_paper/obs_ndispersion.pdf`

Nitrogen dispersion, Eos vs low-a disc, with the high-a (Splash) benchmark.
(a) high-a and (b) low-a samples in [Fe/H]-V_phi (row-normalised density), each
with a low-V_phi box (Splash / Eos) and a disc box over -0.8<[Fe/H]<-0.5; (c)
robust sigma_[N/Fe] (1.48xMAD) vs [Fe/H] for the four bands (low-a solid, high-a
dashed) against the Aurora level (purple band). Shows the low-V_phi N excess is
present in low-a (Eos) but not high-a (Splash). Derived from
`scripts_repro/build_nb.py::disp_figure`; V_tan->V_phi, matched-Delta-sigma
annotations removed, right-panel y starts at 0.05, legend given headroom above
the Aurora line. Portable. Run with the local `astro312`.

## `obs_amr_agedist/` -> `Fig_paper/obs_amr_agedist.pdf`

Eos metallicity-age structure vs the high/low-a disc and Splash (AstroNN ages).
(a) [Fe/H]-[Mg/Fe] with the high/low-a split (solid) and accreted (dashed) lines;
(b) age KDE of Eos / Splash / low-a disc / high-a disc; (c) age-[Fe/H] plane with
nested 90/60/30% KDE contours per population over the grey base density. Same cuts
as all prior analysis: Eos = canonical cut (n=353 -> 191/162; 318 after the
sigma_age/age<0.3 age cut in the age panels), low-a disc = thin_al & V_phi>150,
high-a disc = thick_al & V_phi>150, Splash = thick_al & V_phi<80. Assembled from
`scripts_repro/plot_eos_amr_agedist.py` (itself from `plot_eos_amr.py` +
`build_nb.py` Fig 6). Portable. Run with the local `astro312`.

## `obs_eos_branches_overview/` -> `Fig_paper/obs_eos_branches_overview.pdf`

Five-panel overview of the two Eos branches split by the Davies divider
[Mg/Fe]=0.317*[Fe/H]+0.353. Top row = two large 2D maps: (a) halo [Mg/Fe]-[Fe/H]
density with the accreted (dashed), high/low-a (dotted) and Eos-divider (green)
lines; (b) same plane coloured by mean J_R (RdYlBu_r, red=high J_R, 300-1000
kpc km/s). Bottom row: (c) J_R distribution of the two branches; (d) deconvolved
sigma_[N/Fe] per branch vs [Fe/H] with the low-a disc; (e) age distribution of the
low-a disc by [Fe/H] (YlOrRd, age axis 2-11 Gyr) + the two Eos branches. **Branch->metallicity
mapping is data-driven**: alpha-rich (upper) median [Fe/H]=-0.71 => "Eos metal-poor"
(red); alpha-poor (lower) median [Fe/H]=-0.46 => "Eos metal-rich" (blue). Colours
consistent across (c)/(d)/(e); (e) uses YlOrRd for the [Fe/H] bar. Assembled from scripts_repro
plot_davies_fig2_jr.py + plot_eos_action_dists.py + plot_eos_bifurcation.py +
plot_eos_age_dist.py. Actions via AGAMA/McMillan17 from the AstroNN VAC (Mac-only
6D). Run with the local `astro312`.
