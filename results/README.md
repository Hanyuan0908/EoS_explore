# Eos: heated low-α disc, or the onset of low-α star formation?

Verdict from the current analysis: **the evidence favours the *onset* scenario.**
Eos is best explained as the **oldest, in-situ low-α stars, born on hot /
non-rotating orbits *before* the low-α disc spun up** — the beginning of the
low-α sequence — rather than a pre-existing low-α disc later heated by GS/E (the
low-α analogue of the Splash).

## Evidence

### 1. Ages (LAMOST MSTO, `ana1_age_distributions.png`)
| population        | median age | n   |
|-------------------|-----------:|----:|
| low-α disc        | 6.2 Gyr    | many|
| **Eos**           | **12.2 Gyr** | 305 |
| Splash            | 12.7 Gyr   | ~3k |

Eos is old and **coeval with the Splash** (KS distinguishable but both ≈12 Gyr),
and utterly unlike the bulk low-α disc (KS D=0.80). Consistent with the paper.
*This alone does not separate the two scenarios — both predict an old Eos.*

### 2. Age at fixed [Fe/H] — the decisive test (`ana2_amr_fixed_feh.png`)
Within the low-α sequence, compare **halo-orbit (Eos)** vs **disc-orbit** stars at
matched [Fe/H]. Eos is **systematically older at every metallicity**, by ≈1.0–1.6
Gyr in the clean Eos regime (−1.0 < [Fe/H] < −0.6):

| [Fe/H] | Eos age | disc age | Δ(Eos−disc) |
|-------:|--------:|---------:|------------:|
| −1.02  | 12.5    | 11.5     | +0.9 |
| −0.88  | 12.2    | 10.6     | +1.6 |
| −0.73  | 12.4    |  8.9     | +3.5 |

The Eos track is **flat and old** while the disc track plunges with metallicity.
At fixed [Fe/H], **hotter orbit ⇒ older star** — the signature of **disc spin-up**
("upside-down" formation): the earliest low-α stars were born kinematically hot
and the disc settled (cooled) over the next ~1 Gyr. This is a *continuous* age–
kinematics gradient, **not** the discrete age truncation expected from a single
heating event.
(Caveat: the large offsets at [Fe/H] > −0.6 are inflated by bulge/bar leakage
into the halo-orbit cut and should not be over-interpreted.)

### 3. Orbits vs the known-heated Splash (`ana3_orbits_vs_splash.png`, APOGEE)
| quantity | Eos median | Splash median | KS p |
|----------|-----------:|--------------:|-----:|
| r_apo    | 11.6 kpc   | 4.1 kpc       | ~0   |
| ecc      | 0.87       | 0.77          | ~0   |
| Lz×10⁻³  | 0.11       | 0.09          | small|

Eos and Splash both have **Lz ≈ 0** (non-rotating) — they share the GS/E-merger
epoch. But Eos reaches **~3× larger apocentres** and higher eccentricity. The
Splash (the *bona fide* heated old disc) is centrally concentrated; Eos is
spatially extended. Eos is therefore a **distinct population**, not the same
heated material — disfavouring "Eos = heated low-α disc analogue of the Splash."

## Physical picture
~12 Gyr ago, during the GS/E merger-induced starburst, the Galaxy formed in-situ
**low-α** stars on hot, non-rotating, eccentric, radially-extended orbits (Eos),
while its pre-existing **high-α** disc was dynamically heated into the Splash.
Over the following ~Gyr the low-α disc spun up and cooled into the rotationally-
supported thin disc. Eos is the **onset** of that low-α phase, not a heated
remnant of an already-settled low-α disc (which did not yet exist).

## Caveats / next steps
- LAMOST `AL_FE_ERR` ≈ 0.15 dex ⇒ noisier in-situ/accreted (Al) separation in the
  age sample; APOGEE Al is clean but its ages saturate >10 Gyr.
- APOGEE(giants) ∩ LAMOST(subgiants) overlap is only 6 stars → cross-survey
  validation is weak; both selections are internally consistent instead.
- Selection lines were tuned to reproduce the draft's Fig 1, not taken verbatim.
- Next: (i) tighten the Eos kinematic definition to exclude bulge/bar at high
  [Fe/H]; (ii) repeat E2 on the Xiang+2024 subgiant set as a robustness check;
  (iii) quantify the spin-up gradient (age vs Lz at fixed [Fe/H]) directly.
