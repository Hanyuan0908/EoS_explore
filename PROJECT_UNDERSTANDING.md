# Eos — project understanding & story

Working synthesis of what the project shows, the interpretation we've agreed on, and
the confidence level attached to each claim. Written for paper-drafting: the "what's
solid vs inference vs speculative" tiers at the end are the operative part.

Companion docs: `Fig_code/README.md` (what each figure is + how it's made),
`Fig_code/CONVENTIONS_observational.md` (cuts, colours, data provenance), and the
draft `Non_rotating_low_alpha.pdf`.

**Naming:** the population is **Eos** everywhere in the figures and this project. The
draft text still calls it *"Aura"* in places — legacy, to be globally replaced by Eos.

---

## 1. One-paragraph story

Eos is an **in-situ, low-α population on halo-like orbits** (near-zero V_φ, high
eccentricity) — distinct from the ordered low-α thin disc despite sharing the low-α
chemistry. Its Mg/Al chemistry and its age both sit *between* GS/E and the low-α disc.
It is the chemical/kinematic fingerprint of the **GS/E-merger-induced starburst** that
marks the onset of low-α disc formation. Eos is **not a single object**: it is a
**composite** of (i) a *splashed/heated pre-existing low-α disc* component and (ii) a
*lane-born halo* component formed from poorly-mixed gas in the tidal bridge during the
merger — both triggered by the same GS/E pericentre passage, and separable in the data
as the metal-rich and metal-poor branches.

---

## 2. Data & sample

- APOGEE DR17 (allStarLite abundances) + AstroNN VAC (kinematics, actions, ages),
  Gaia EDR3 astrometry, MWPotential2014. Belokurov & Kravtsov 2022 (B&K22) cleaning;
  red giants, d_helio < 15 kpc. Base sample ≈ 529k.
- Actions (J_R, L-vector) via AGAMA/McMillan17 from the AstroNN 6D VAC (Mac-only file).
- Ages: AstroNN default; Anders 2023, BINGO (Ciucă 2024), LAMOST/isochronal for
  cross-catalogue checks. Age-quality cut everywhere: finite, 0<age<20 Gyr,
  σ_age/age < 0.3.

---

## 3. The chemical machinery & the canonical Eos cut

Three populations carved from chemistry, refined with [Al/Fe] (Fig 1–2 of the draft):
- **accreted (GS/E):** low [Al/Fe] ≲ −0.1, fast Mg decline (Al suppressed in dwarfs).
- **high-α:** early disc + Aurora (pre-disc, [Fe/H]<−1) + Splash.
- **low-α:** late disc, formed after the GS/E interaction; lowest [Mg/Fe] at high
  [Fe/H] but *intermediate* α at low [Fe/H].

**Reproducibility anchor — any Eos figure must give n = 353 (191 / 162):**
```
halo    = base & ((ecc>0.7) | (lz<0))                 # ecc=(rap-rperi)/(rap+rperi)
acc_line= -0.30*feh - 0.10                            # accreted / in-situ
hl_line = -0.14*feh + 0.135                           # high-α / low-α
divider = 0.317*feh + 0.353                           # Davies α-rich / α-poor split
lowa    = halo & (-0.9<feh<-0.2) & (mg>acc_line) & (mg<hl_line) & (al>-0.12)  # n=353
eos_hi  = lowa & (mg >  divider)   # α-rich / UPPER  n=191  median [Fe/H]=-0.71
eos_lo  = lowa & (mg <= divider)   # α-poor  / LOWER n=162  median [Fe/H]=-0.46
disc    = thin_al  & (galvt>150)   # low-α disc reference
splash  = thick_al & (galvt<80)    # Splash reference
```
Note the branch↔metallicity mapping is the *opposite* of a naive rename and is
data-verified: **α-rich/upper = "Eos metal-poor"; α-poor/lower = "Eos metal-rich".**

---

## 4. The two-branch mapping (the headline result)

Same split, seen consistently in simulation and data:

| | metal-poor branch | metal-rich branch |
|---|---|---|
| chemistry | α-rich / upper, median [Fe/H]=−0.71 | α-poor / lower, median [Fe/H]=−0.46 |
| J_R | higher (~690 kpc·km/s) | lower (~530) |
| σ_[N/Fe] | elevated, Aurora-like at metal-poor end | tracks the low-α disc |
| age | older (peak ~7.5 Gyr) | younger, but still older than co-[Fe/H] disc |
| sim analogue | **born-hot** (larger R_birth, metal-poor, larger J_R) | **born-cold** (smaller R_birth, metal-rich, smaller J_R) |
| origin | **lane-born halo** (GS/E tidal bridge, unmixed gas) | **splashed / heated low-α disc** |

The sim→data cross-match (born-hot/born-cold ↔ metal-poor/metal-rich, same J_R and
σ_N trends) is a genuine prediction-then-confirmation, not a post-hoc label.
Separations are **distributional tendencies with substantial overlap**, not clean
splits — but the *direction* is robust and the overlap is comparable in sim and data.

---

## 5. Evidence, figure by figure

**Observations (`Fig_paper/obs_*`):**
- `obs_energy_pops` (E–Lz): Eos is a tight |Lz|≈0 clump at E×10⁻⁵≈−0.55, disconnected
  from both the accreted cloud and the disc sequence. → *kinematically its own thing.*
- `obs_alfe_pops`, `obs_mg_al_meanal`: Eos chemistry intermediate between GS/E and the
  low-α disc; recovered even from a **kinematic** cut (not just the chemical wedge),
  so the intermediate-ness is real, not selection-induced.
- `obs_amr_agedist`: Eos age intermediate — older than the low-α disc, **matching
  Splash** (both are GS/E's children).
- `obs_nfe_pops`: P5/P95 [N/Fe] tracks show the **asymmetric, upward (high-N tail)**
  spread of Eos — the shape argument.
- `obs_ndispersion`: quantified σ_[N/Fe] with clean errors; low-α Eos (V_φ<75) sits
  **systematically above the co-metallicity disc**, reaching the **Aurora** level
  (0.149±0.004, same symmetric σ) at the metal-poor end — and this excess is **absent
  in high-α (Splash)**, so it is Eos-specific, not generic heating. This is the single
  strongest argument that Eos is not merely a splashed, well-mixed disc.
- `obs_eos_branches_overview`: ties the two-branch story together (J_R, σ_N, age per
  branch).

**Simulations (`Fig_paper/au18_*`, `splash_*`):**
- `au18_birth_positions_gas4`: halo-born stars trace the **tidal lane/bridge** joining
  GS/E and the host during the merger (t=4.99 Gyr).
- `au18_gas_metallicity`: the lane gas is **intermediate [Fe/H]** (MW×GS/E mixing) —
  explains why the *lane-born (metal-poor)* branch has intermediate chemistry.
- `au18_birth_orbits`: total & halo-born SFR **peak at the GS/E pericentre**, and the
  halo-born/disc-born ratio **spikes** there → the pericentre re-triggers halo-orbit
  star formation. (See caveat 6b.)
- `au18_vr_vphi_three`: applying the Eos selection at z=0 recovers **two birth
  populations** — born-cold and born-hot — i.e. Eos is composite.
- `au18_rbirth_feh_jr`: born-cold = smaller R_birth, more metal-rich, smaller
  present-day J_R (disc-like); born-hot = the opposite (halo-like).
- `splash_vphi_evolution_3panel`: in GASTRO a splashed low-α disc floors at V_φ ~55–60
  km/s and **cannot reach Eos's observed V_φ ≈ 0** — disfavours a *pure* splash origin
  for (all of) Eos.

---

## 6. Interpretation & the agreed caveats

**Scenario framing.** The two origin scenarios — (1) low-α "Splash" (heated disc) and
(2) GS/E-lane-born halo burst — are **not either/or**. Eos is **both**, split by branch:
metal-rich = (1), metal-poor = (2). Lead with this composite picture.

**N-dispersion physics.** High σ_[N/Fe] = signature of **unmixed / stochastic**
enrichment (well-mixed gas homogenises abundances, leaving no scatter). Eos shares
Aurora's *chemical-inhomogeneity signature* — **not its epoch** (Eos forms later, at the
merger, at higher [Fe/H]). Present the chemistry (N tail → unmixed gas) and the
kinematics (halo orbit) as **two independent legs**, both symptoms of the unsettled
lane — not a causal chain "stochastic SF → halo orbits."

**Wording fixes flagged for the draft:**
- (a) **Ages are used relatively**, to order events (Eos between high-α and low-α); they
  are *not* an absolute GS/E clock (AstroNN absolute scale is unreliable >10 Gyr / below
  [Fe/H]=−1). Deliverable: report the Eos age in **each** catalogue (AstroNN / Anders /
  BINGO / isochronal) so each community can anchor the transition epoch on its own scale.
- (b) **"halo starburst only at the pericentre" is too strong** — `au18_birth_orbits`
  panel (d) shows the halo-born/disc-born ratio is *higher* pre-disc-spin-up. Correct
  statement: the pericentre is the **only halo-born burst *after* disc spin-up**, and the
  only one at low-α disc metallicities.
- (c) **"oldest low-α disc was splashed → simultaneous formation"** is the most exposed
  claim (stacked soft inferences × the age-cut-thinned old branch, N ~a dozen outside
  AstroNN). Soften "older than the *oldest* disc" → "older than the *bulk* disc," label it
  an **inference**, and state the surviving N per bin. Drop "tidally disrupted" →
  "dynamically heated."
- (d) Quantify branch overlap and median offsets in **both** sim and data side-by-side —
  the *consistency* of the overlap is the evidence, so show it numerically.

**Closed / do not re-open:** the `GRIDEDGE_WARN` per-element-flag question — checked, the
low-N stars are near but not on the grid edge (no pile-up), no artifact; B&K22 use only
global quality flags. The published N figures need no correction for it.

---

## 7. Confidence tiers

**Solid (defensible to a referee as-is):**
- Eos is kinematically distinct (E–Lz clump, V_φ≈0, high ecc).
- Intermediate chemistry (real, recovered from a kinematic cut) and intermediate age
  (Eos older than the low-α disc, coeval with Splash).
- σ_[N/Fe] systematically above the co-metallicity disc, Aurora-like at the metal-poor
  end, and Eos-specific (null in Splash).
- The Au18 lane → intermediate lane metallicity → pericentre SFR/halo-fraction peak chain.
- The two-branch composite and its sim↔data cross-match (as distributional tendencies).

**Inference (state as such):**
- metal-rich = splashed low-α disc; metal-poor = lane-born halo.
- Eos as the GS/E-triggered onset of the low-α disc.

**Speculative (weakest link, flag explicitly):**
- Low-α disc formation *simultaneous* with (not after) the merger, inferred from the
  metal-rich branch being older than the bulk disc — rests on the small, age-cut-thinned
  old branch and the relative age ordering.
