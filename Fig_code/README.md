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

## `au18_nitrogen_dispersion/` -> `Fig_paper/au18_nitrogen_dispersion.pdf`

Chemical inhomogeneity of the two birth classes at the GS/E pericentre, and where
it comes from.  Stars formed in the snapshot-72 window (t_form = 4.83-4.99 Gyr),
split by the same criterion as `au18_birth_orbits`.  (a) the halo-born stars
edge-on, coloured by [N/Fe], with the two apertures drawn; (b) sigma_[N/Fe]
against [Fe/H], disc-born against halo-born; (c) the halo-born curve split by
birth site.

Three apertures, measured in the frame of `au18_birth_positions_gas4` with the
azimuth pinned by `au18_frame.align_azimuth`:

    GS/E analogue   |r - r_GSE| <  6 kpc   (r_GSE = clean-debris centroid)
    MW analogue     |r - r_GSE| >= 6 and |r| < 8 kpc
    bridge          everything else

Counts, which a rerun must reproduce: 26,064 disc-born and 12,608 halo-born, the
latter splitting 2,042 / 6,886 / 3,680 into GS/E analogue / bridge / MW analogue.
The same 26,064 and 12,608 appear on `au18_birth_positions_gas4`, so they are the
fastest check that the sample is right.

A bin is drawn only where it holds **150 stars**; below that the bootstrap band is
wider than the differences the figure exists to show.  That drops the
[Fe/H] < -1.0 bin from every curve, the GS/E and MW analogues below -0.8, and
everything above +0.2 except the two halo-born curves.

Panels (b) and (c) are 6:4 and share an axes height with (a), which is square
because `aspect='equal'` over a 50 x 50 kpc frame.  The three rects are placed
explicitly in inches -- `tight_layout` will not honour a mixed-aspect row, and
leaves (a) shrunken inside an oversized slot.

**Fixed [Fe/H] bins are not cosmetic.**  The two classes differ by 0.61 dex in
median metallicity and sigma_[N/Fe] runs steeply with [Fe/H], so comparing them
unbinned returns that gradient as if it were a difference between the classes.

The result: the dispersion belongs to the BRIDGE, not to the halo orbit.
Halo-born stars formed inside either compact aperture sit near 0.016 dex at every
metallicity; only those formed in the lane reach 0.057.  Panel (a) shows why --
GS/E-like and host-like gas occupy opposite ends of the lane and have not mixed.

Two things to state in the caption:

- **The colour scale in (a) is diverging about [N/Fe] = 0.09**, which is the
  measured divider between the two chemical branches, not an arbitrary midpoint.
  cmasher's `iceburn` was tried and rejected: its black midpoint swallowed whole
  panels, since most of this sample sits near the divider.
- **Auriga's abundances are not the Milky Way's.**  Every dispersion in the figure
  is 0.01-0.06 dex, against the ~0.3 dex separating the MW's two alpha sequences.
  These are ISM inhomogeneities within the model and must not be set beside an
  observed sigma_[N/Fe].  The [Mg/Fe] zero point in this run is offset the same
  way (see `auriga/ana_eos_mgfe_feh.py`).

The chain is the `au18_birth_orbits` one (`prep_birth_actions.py`, `prep_zmax.py`
and the potentials behind them, all symlinked here); the figure script reads
snapshot 72 directly for `GFM_Metals`, plus `out/gse_clean_ids.npy`.

`au18_frame.py` is kept here as a REAL COPY, not a symlink into `../../auriga/`.
The `main` branch tracks only `Fig_paper` and `Fig_code`, so a link reaching
outside `Fig_code` resolves on this branch and dangles on `main` -- which is what
has happened to `Fig_code/FINDINGS.md` and to `au18_gas_metallicity/orbit_tools.py`.
Anything `Fig_code` needs must live inside `Fig_code`.  Runs in a
few minutes with the local `astro312`.

The diagnostics behind it, none of which are paper figures, are
`auriga/diag_nfe_dispersion_origin.py` (site decomposition and the provenance
comparison against the clean GS/E debris), `diag_nfe_bridge_trend.py` (the
2-Gaussian decomposition of the bridge trend), `diag_nfe_bimodality.py` (what the
two components are, element by element) and `diag_nfe_map_feh.py` (the same map
as panel (a), split into metallicity bins).

---

## `au18_nitrogen_map/` -> `Fig_paper/au18_nitrogen_map.pdf`

The companion map to `au18_nitrogen_dispersion`: the same halo-born sample, one
panel per 0.2 dex bin of [Fe/H], edge-on, coloured by [N/Fe] over the snapshot-72
gas surface density.  Shows *why* the dispersion belongs to the bridge -- in every
bin where both compositions exist they occupy opposite ends of the lane and have
not mixed.

Panels abut exactly (shared axes, labelled once) and carry no annotation but the
metallicity bin.  The counts, which explain the differing opacity and belong in
the caption, are, from the most metal-poor bin up:

    112, 1,178, 2,435, 3,777, 2,844, 1,419, 637, 184

**The 150-star threshold of `au18_nitrogen_dispersion` is deliberately NOT applied
here.**  That cut exists because a dispersion from a hundred stars has a bootstrap
band wider than the effect; a map of a hundred positions is perfectly legible, and
the sparse [Fe/H] < -1.0 panel is the one showing a single composition filling the
whole frame, which is the point the figure makes.

Opacity is `clip(900/N, 0.20, 0.95)` per panel rather than a fixed value: N spans
a factor of 34 across the bins, and at one opacity the crowded intermediate panels
saturate into a block that hides the interface between the two streams.

The rects are placed explicitly in inches.  With `wspace = hspace = 0` and
`aspect='equal'`, any mismatch between slot shape and data shape reopens the gaps,
and no gridspec setting fixes it.

Caption must state the same two things as the companion figure: the [N/Fe] scale
is diverging **about the measured branch divider at 0.09**, not an arbitrary
midpoint; and Auriga's [N/Fe] range here is a tenth of what separates the Milky
Way's alpha sequences, so these are model ISM inhomogeneities only.  The gas scale
is stretched to the disc-to-lane transition, not to the full range of the frame.

Same chain and environment as `au18_nitrogen_dispersion`; `au18_frame.py` is a
symlink into that directory, which holds the real copy.

---

## `au18_birth_classes/` -> `Fig_paper/au18_birth_classes.pdf`

`au18_birth_positions_gas4` and `au18_birth_orbits` merged into one four-panel
figure, so the spatial and temporal halves of the birth-class result sit together.
(a), (b) the two classes at the GS/E pericentre over the gas surface density, with
the clean-debris contours; (c) the star-formation history split by class;
(d) their ratio.  Same classification as `au18_birth_orbits`.

Counts a rerun must reproduce: 1,932,134 in-situ stars with a birth orbit, 8.6 per
cent halo-born; 26,064 disc-born and 12,608 halo-born in the snapshot-72 window;
peak halo/disc SFR ratio 0.46 at t = 4.96 Gyr.

**Dropped from the parent figures**, by request: the eps-against-time map and the
eps distributions by epoch (both carried the classification, which this figure
takes as given), and the rotated epoch label on the right edge of the maps, which
made the top panels a different width from the bottom ones.  The epoch is stated
inside panel (a) instead.  The parent figures are kept; this is an addition.

**The panel rectangles are placed explicitly in inches.**  The top row is forced
to `aspect='equal'` and the bottom row is not; `tight_layout` will not keep a
mixed row aligned -- it shrinks the equal-aspect axes inside their slot and
centres them, which is the misalignment this figure exists to avoid.  The same
trap is recorded under `au18_nitrogen_map`.

Carried over from the parents and still true here: the SFR kernel width `BW_T` is
a Gaussian **sigma**, not a bin width -- 0.15 would oversmooth the narrow
halo-born spike and drag the panel-(d) ratio down by a third; and the scatter
opacity in (a)/(b) is scaled to each panel's N, which differ by 2x.

`orbit_tools.py` and `au18_frame.py` are kept inside `Fig_code` (a real copy here
and a link to `au18_nitrogen_dispersion` respectively) rather than linked out to
`../../`, because `main` tracks only `Fig_paper` and `Fig_code` and an outward
link dangles there.

Chain and environment as `au18_birth_orbits`; the figure script additionally reads
snapshot 72 directly for the maps.  A run takes a few minutes with the local
`astro312`.

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

### `fig_paper_nfe_pops_3pop.py` -> `Fig_paper/obs_nfe_pops_3pop.pdf` (+ `obs_cfe_pops_3pop.pdf`)

Same figure/style as `obs_nfe_pops` but 2x3, with the **accreted** column re-added on
the left; the two in-situ panels are identical to `obs_nfe_pops`. The accreted panel
carries an extra chemical outlier rejection (N/Fe figure only): accreted stars above
the line (-1.5, 0.6)->(-0.5, 0.42) or below (-1.5, -0.2)->(-1.0, -0.4) are removed
(green dashed lines drawn; 193 stars, 2572->2379). These are chemical outliers only --
the `acc` mask already applies |Lz| < 500 and every accreted star has a finite V_phi.
Shared greyscale stretch across all three top panels. Motivated by B&K22 (Belokurov &
Kravtsov 2022): after the cut the accreted [N/Fe] spread is smaller than the in-situ
Aurora, as B&K22 report, without damping the Aurora dispersion. Portable. Run with the
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

## `obs_eos_age_dist/` -> `Fig_paper/obs_eos_age_dist.pdf`

Two-panel age distributions of Eos and its two metallicity branches vs Splash and
the low-a disc, in two age catalogues: (a) AstroNN (`age`, sigma_age/age<0.15);
(b) BINGO (age = 10**`pred_logAge`; Ciuca+2024 cuts with sigma(log tau)<=0.1,
0-18 Gyr raw scale). Curves are Gaussian KDEs. Eos = canonical cut (n=353 ->
191/162); the two branches are split by the Davies divider
[Mg/Fe]=0.317*[Fe/H]+0.353 and relabelled by their data-driven metallicity:
alpha-rich/upper => **Eos, metal-poor** (tomato); alpha-poor/lower => **Eos,
metal-rich** (blue). Splash = thick_al & V_phi<80; low-a disc = thin_al & V_phi>150.
One shared legend (identical for both panels), no counts/annotations on the figure.
The alpha-rich/metal-poor branch is systematically older than the alpha-poor/metal-rich
branch in both catalogues; BINGO branch counts are modest (23/23) so treat those
dashed curves with care. Assembled from `scripts_repro/plot_eos_age_dist_pub.py`
(family: `plot_eos_age_dist_4cat_branches.py`). BINGO ages are Mac-only
(`APOGEE_DR17_bingoages.fits`). Run with the local `astro312`.
