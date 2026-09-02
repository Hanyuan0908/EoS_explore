# Au18 / GS/E: what we have established

Results, with the numbers needed to check them and the caveats that go with them.
Method conventions for figures are in `../Fig_code/CONVENTIONS.md`; the z_max
approximation is in `METHOD_zmax_from_Jz.md`.

Unless stated otherwise the sample is in-situ stars of Au18 and the merger window
is t_form = 4.99-6.54 Gyr.

---

## 1. The merger and the starburst coincide

| quantity | value |
|---|---|
| GS/E first apocentre | t ~ 3.0-3.5 Gyr, r ~ 215 kpc |
| pericentre plunge | t ~ 5.0 Gyr |
| coalescence / phase-mixing | t = 5.3-5.6 Gyr (z ~ 1.1-1.2) |
| in-situ SFR peak | **25.8 Msun/yr at t = 5.15 Gyr** (100 Myr bins; 19.2 smoothed) |
| baseline SFR (3.0-4.5, 7.0-9.0 Gyr) | 11.4 Msun/yr |
| burst amplitude | **1.7x baseline** |
| GS/E progenitor's own SF truncation | t = 3.22 Gyr (90th pct) |
| in-situ mass formed in the window | 2.4e10 Msun = 16.3 per cent |
| in-situ mass already in place at coalescence | 39 per cent (32 per cent at pericentre) |

The SFR peak sits 0.3 Gyr *before* the centre of the coalescence window -- between
the plunge and full phase-mixing, where a gas-compression burst is expected.

**Caveats.**  1.7x is a factor-of-two enhancement, not an order of magnitude; the
SFH is bumpy throughout, with secondary peaks near 14.5 Msun/yr at t ~ 2.5, 9 and
10.5 Gyr.  The timing is the strong part of the argument, not the amplitude.  The
progenitor stopped forming stars ~2 Gyr before coalescence, so no GS/E star can
be leaking into the in-situ burst.

## 2. The burst preferentially makes hot orbits

Birth circularity eps = L_z/L_circ(E) in each epoch's own AGAMA potential and disc
frame.  With eps < 0.5:

| epoch | halo-born mass fraction |
|---|---|
| quiet before, 3.5-4.7 Gyr | 8.1 per cent |
| burst, 4.9-5.7 Gyr | **19.2 per cent** |
| quiet after, 6.6-8.0 Gyr | 14.0 per cent |

At the peak the halo-born SFR rises **x4.97** while disc-born rises only x1.82 --
the burst is close to a pure hot-orbit event superimposed on a normal disc.

**The eps distribution is unimodal at every epoch**, at birth and at z = 0, with no
interior minimum.  Any cut is therefore a convention and the absolute fraction is
meaningless without quoting it (4 per cent at eps<0.2, 50 per cent at eps<0.8).
What survives the choice is the **ratio**, flat at 2.3-2.4 across cuts from 0.2 to
0.8 with the AGAMA circularity.  Cut-free version: the 10th percentile of
eps_birth drops 0.448 -> 0.279 -> 0.325 across the three epochs while the median
barely moves -- the burst thickens the low-eps tail, it does not shift the disc.

## 3. Circularity alone does not separate halo orbits from BAR orbits

The most important methodological result.  Bar orbits are planar and elongated
with low L_z for their energy, so they fall below any circularity cut.  Measured
by the m=2 Fourier mode of the existing disc:

| snapshot | t | bar A2/A0 | bar PA | inner "halo-born" PA | offset | b/a |
|---|---|---|---|---|---|---|
| 68 | 4.36 | 0.108 | 166.0 | 173.2 | +7.2 | 0.66 |
| 80 | 6.21 | 0.231 | 104.6 | 103.7 | **-0.9** | 0.45 |
| 100 | 9.41 | 0.405 | 25.5 | 27.7 | **+2.2** | 0.36 |
| **72** | **4.99** | **0.088** | 177.9 | 162.2 | **-15.8** | **0.87** |

At snapshots 80 and 100 the low-eps newborns *are* the bar -- aligned to within a
couple of degrees, b/a ~ 0.4, centrally concentrated (median R_birth 1-2 kpc).
**Snapshot 72, the merger peak, is not**: round (b/a = 0.87), 16 degrees off, born
at median R = 5.4 kpc and genuinely off-plane (median |z| = 3.5 kpc, RMS 11.9).
The bar strengthens monotonically after the merger (peak A2/A0 0.15 -> 0.45) and
is 6-9 kpc long, so an R > 3 kpc cut does not escape it.

**Fix**: require a halo-born star to be vertically extended as well as slowly
rotating.  With eps <= 0.8 AND z_max >= 1.5 kpc:

| epoch | halo-born |
|---|---|
| before | 6.5 per cent |
| **during** | **22.8 per cent** |
| after | 5.2 per cent |

**3.5x over before, 4.4x over after**, and the post-merger value falls back below
the pre-merger one -- the signal is transient.  Instantaneous |z| works too but is
phase-dependent; z_max is not.  See `METHOD_zmax_from_Jz.md` for its accuracy.

## 4. Heating dominates settling 5:1

Burst cohort (207,654 stars), eps cut 0.5, birth vs z = 0:

| | N | per cent of its birth class |
|---|---|---|
| born disc -> still disc | 109,163 | 65.3 |
| born disc -> now halo | **58,105** | 34.7 |
| born halo -> now disc | **11,050** | 27.4 |
| born halo -> still halo | 29,336 | 72.6 |

Net flow out of the disc class is +47,055, a ratio of 5.3 : 1.  An earlier
34.5 per cent halo->disc rate was largely an artefact of the 95th-percentile
envelope estimator, which failed in the central few kpc (those stars sat at
median R_birth = 2.17 kpc against 3.88 for everything else); with the AGAMA
potential the anomaly is gone (4.08 vs 3.98 kpc).

Both off-diagonal cells are still inflated by the cut sitting on the steep flank
of a unimodal distribution.  Cut-free: median d(eps) = -0.10 for the burst cohort,
63.5 per cent decreasing, only 36.8 per cent moving by |d eps| > 0.3.

## 5. The disc reorients through the merger -- and it is physical

Angle of the disc angular-momentum axis from its z = 0 direction:

| t [Gyr] | 4.36 | 4.99 | 5.59 | 6.21 | 8.10 | 13.35 |
|---|---|---|---|---|---|---|
| angle | 103 | **94** | 40 | 25 | 11 | 2 |

Nearly perpendicular at the burst.  This is **not** a bookkeeping artefact: a star
that keeps its angular momentum while the gas disc reforms in a new plane really
is on an inclined orbit relative to today's disc, and one born aligned with the
plane the disc is *about to* adopt really was born on a disc orbit of the future
disc.  Measuring each epoch in its own frame is the correct treatment.

## 6. The halo-born stars form in the gas lane joining GS/E to the host

At t = 4.99 Gyr the GS/E centroid is at (x, z) = (-10.7, -14.4) kpc, r = 20 kpc,
and the halo-born stars trace a continuous lane from the disc into it, following
the gas bridge rather than filling the halo.  By t = 9.41 Gyr the gas is a thin
disc again and the few halo-born stars are structureless.

**The lane is the host's gas on a metallicity gradient, not a second reservoir:**

| | disc | lane | GS/E |
|---|---|---|---|
| gas [Fe/H] | -0.29 | -0.48 | -0.69 |
| sigma([Fe/H]) | 0.28 | 0.21 | 0.24 |
| [O/Fe] | +0.25 | +0.26 | +0.25 |
| [Mg/Fe] | -0.31 | -0.27 | -0.27 |
| sigma([X/Fe]), all metals | 0.01-0.05 | 0.01-0.05 | 0.01-0.05 |

[Fe/H] falls 0.4 dex continuously with no discontinuity, while every [X/Fe] is
flat to within 0.06 dex and its scatter is an order of magnitude below the
scatter in [Fe/H].  Two reservoirs meeting would inflate sigma([X/Fe]) where they
overlap; nothing like that appears.

**Caveat.**  Gas provenance is not tracked.  By pericentre the satellite's own gas
may already be stripped, so "the GS/E region" may be measuring host gas sitting at
the satellite's position.  Settling that needs gas cells traced back to the
progenitor, which has not been done.

Note also: Mg sits ~0.3 dex below the other alpha elements in Auriga's yield set
(median [Mg/H] = -0.95 vs [O/H] = -0.42), so [Mg/Fe] and [O/Fe] will not agree
with each other or with observations.

## 7. The two Eos populations at z = 0

Split at v_phi,birth = 150 km/s within the Eos cut (|v_phi| < 80, ecc > 0.6):

| | born-cold (3,300) | born-hot (4,283) |
|---|---|---|
| median R_birth | 4.23 kpc | **10.37 kpc** |
| median [Fe/H] | -0.16 | -0.40 |
| median J_R, all | 336 | 986 |
| median J_R, solar neighbourhood | 446 | 752 |

The Eos selection takes a horizontal slice out of one rotating distribution at
z = 0, but at birth those stars are two groups -- one on the disc ridge near
+220 km/s, one already slow -- with the 150 km/s split falling in the dip between
them.  sigma_R of the Eos subset is *higher* at birth (141) than at z = 0 (127):
the radial motions were already large when they formed, so "heated" describes the
born-cold half, not the selection as a whole.

**A solar-neighbourhood selection compresses the difference** (J_R 336 -> 446 for
born-cold, 986 -> 752 for born-hot), so the two populations look more alike
locally than they are globally.  That is an observational-bias statement worth
making explicitly.

Eos-like stars are drawn preferentially from the *early* part of the window --
median t_form 5.28 Gyr against 5.75 for all merger-born stars.

## 8. Corrections that changed published numbers

| what | before | after |
|---|---|---|
| eps > 1 (unphysical), at birth | 11.6 per cent | **0.00 per cent** |
| burst/quiet halo-born ratio | 1.77, drifting 1.96->1.17 with the cut | 2.37, flat across cuts |
| halo->disc transitions | 34.5 per cent | 27.4 per cent |

Causes: the original estimator divided j_z by the 95th percentile of j_z among
prograde stars in each energy bin -- an empirical envelope normalised by the star
distribution, not the potential, so ~5 per cent of prograde stars exceed it by
construction and the normalisation drifts as the galaxy settles.  Replaced by
L_z/L_circ(E) against an AGAMA CylSpline fitted to the particle distribution.

Building those potentials has two requirements that are easy to miss: exclude the
low-res boundary DM (types 2, 3) and keep the grid at Rmax = 50 kpc.  Including
them and stretching to 400 kpc starved the centre of resolution, gave an outward
radial force inside 0.5 kpc, and made AGAMA's ActionFinder refuse to initialise.

## 9. Sample purity

The in-situ sample has **zero** overlap with the ex-situ ID list and **zero** with
the 21,487 clean GS/E IDs.  632,249 star particles are unclassified by the
provenance catalogue, but their median radius is 4.7 Mpc and **none** lies inside
the 33.8 kpc galaxy aperture.  No accreted contamination.

## 10. Open questions

- **Gas provenance.**  Whether the lane gas is host or stripped-satellite material
  is unresolved (section 6).  Needs gas cells traced to the progenitor.
- **The residual halo->disc channel.**  27 per cent of born-hot stars end on disc
  orbits.  Partly boundary scatter, but plausibly also real: stars born misaligned
  with the old stellar disc but aligned with the plane it was about to adopt.
- **The disc axis is defined from stars within 10 kpc**, i.e. the *old* stellar
  disc.  Around coalescence the star-forming gas may already sit in a different
  plane, and that is the plane stars are actually born into.  Redefining the axis
  from the gas would test it.
- **Non-axisymmetric potentials.**  Everything uses axisymmetric CylSpline, so the
  bar is averaged out of the orbit calculation.  Doing better needs the pattern
  speed, which cannot be read off the bar position angle at 0.15 Gyr snapshot
  spacing because the bar turns more than once between snapshots.
