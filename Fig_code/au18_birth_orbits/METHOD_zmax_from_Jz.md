# Approximating z_max from the vertical action J_z

How the birth-orbit classification gets a vertical excursion for every star
without integrating 1.9M orbits. Written down because the normalisation is easy
to get wrong and the accuracy limits are not obvious from the code.

Implemented in `prep_zmax.py`; validated by `diag_jz_to_zmax.py`.

## The relation

For motion at fixed guiding radius `Rg` in a static axisymmetric potential, the
vertical action is the area of the vertical phase-space loop divided by 2*pi:

    J_z = (1/2pi) * closed-loop integral of v_z dz

The loop runs -z_max -> +z_max -> -z_max, which is **four times** the quarter
integral from 0 to z_max, so

    J_z = (2/pi) * Integral_0^z_max sqrt( 2 [ Phi(Rg, z_max) - Phi(Rg, z) ] ) dz

**The (2/pi) prefactor is the one that matters.** Using (1/pi) is a factor-2
error in J_z which, inverted, inflates z_max by ~37 per cent for near-circular
orbits. It is easy to miss because it partially cancels against the
approximation error below: with (1/pi) the overall median z_max is only 15 per
cent high, which looks acceptable and is not. The two errors were separated by
splitting the validation by radial action -- in the near-circular limit the
approximation is exact, so only the correct normalisation converges to 1 there.

    J_r/(J_r+|J_phi|) < 0.02      (1/pi): 1.368       (2/pi): 0.924

`J_z` is monotonic in `z_max` at fixed `Rg`, so the relation inverts uniquely.

## Implementation

1. `Rg = pot.Rcirc(L=|J_phi|)` from the star's own birth potential.
2. Tabulate `J_z(Rg, z_max)` on a log grid, 56 x 80 points, `Rg` = 0.3-48 kpc and
   `z_max` = 0.02-40 kpc. The integral uses the substitution `z = z_max sin(u)`,
   which removes the square-root singularity at the turning point (the integrand
   becomes smooth, going as `(pi/2 - u)^2`), with 48-point Gauss-Legendre in `u`.
3. Invert each `Rg` row onto a log grid in `J_z` to get `z_max(Rg, J_z)`, then
   bilinearly interpolate in `(log Rg, log J_z)` for all stars at once.
4. Below the table, vertical motion is harmonic, so `J_z ~ z^2` and
   `z_max = z_grid[0] * sqrt(J_z / T[0])`. Above it, clip at 40 kpc.

Each star uses the potential nearest in time to its birth snapshot
(`out/potentials_ref/`), so the conversion is automatically epoch-appropriate:
one fixed threshold in kpc, no per-epoch recalibration, even though the potential
deepens from v_c(8 kpc) = 165 to 258 km/s over the run.

## Accuracy

Validated at snapshot 80 against direct orbit integration in the same static
potential (`agama.orbit`, 40 circular periods -- converged: the median z_max
changes by 0.04 per cent between 10 and 40 periods and by 0.03 per cent more out
to 1000; sampling density is irrelevant, identical at 100, 500 and 1500 samples
per period). 1422 orbits:

| J_r/(J_r+abs(J_phi)) | median z_max(action) / z_max(orbit) |
|---|---|
| < 0.02   circular  | 0.92 |
| 0.02-0.05          | 0.85 |
| 0.05-0.15          | 0.77 |
| > 0.15   eccentric | 0.57 |

**This underestimates z_max**, mildly for circular orbits and by up to ~40 per
cent for eccentric ones. The cause is fixing `R = Rg`: an eccentric orbit climbs
higher near apocentre, where the vertical restoring force is weakest, and the
fixed-Rg vertical potential does not know that. The error is therefore worst for
exactly the radially eccentric orbits the halo-born class is made of.

Consequences:

- A nominal cut at `z_max > 1.5 kpc` behaves like a somewhat **stricter** cut on
  the true excursion, more so for eccentric stars. That biases against the
  merger population, so the merger enhancement it measures is conservative.
- It is monotonic in `J_z` at fixed `Rg`, so it ranks orbits correctly. That is
  all a classifier needs.
- **Do not quote these z_max as physical heights.** For a real number, integrate:
  `agama.orbit` runs at ~700 orbits/s, so the full sample costs ~45 min.

The epicyclic shortcut `z_max = sqrt(2 J_z / nu)` is the small-amplitude limit
and is much worse -- median 0.67 overall, 0.39 for orbits reaching beyond 5 kpc.

## Not covered

- The potentials are **axisymmetric** (`symmetry='axisymmetric'`, mmax=0). The
  bar is averaged away. `compute_auriga_potential.py` also builds a `symmetry='n'`
  version; a non-axisymmetric treatment would additionally need the pattern speed
  and a rotating frame, and the pattern speed cannot be read off the bar position
  angle at 0.15 Gyr snapshot spacing because the bar turns more than once between
  snapshots. Not attempted.
- Actions come from AGAMA's Staeckel fudge, not exact.

## Frame convention -- a trap

`ap.util.rotateto` puts the disc axis on component 0. Mapping it to z with
`(c[:,2], c[:,1], c[:,0])`, as `compute_auriga_potential.py` does, is a
**transposition of axes 0 and 2: determinant -1, a reflection.** That is harmless
when only a density is being fitted, but angular momentum is a pseudovector, so
it flips the sign of L_z and puts the whole disc at eps = -1. Use the cyclic
`(c[:,1], c[:,2], c[:,0])`, determinant +1, whenever kinematics are involved.

Under a reflection L_z -> -L_z while J_r, J_z, E, R and z are invariant, so the
correction to an already-computed file is exact: negate `eps`, `Lz` and `Jphi`.

## Reproduce

    python prep_potentials_ref.py        # 36 CylSpline potentials, ~25 min
    python prep_birth_actions.py         # eps, J_r, J_z, J_phi, ~15 min
    python prep_zmax.py                  # -> out/birth_orbits_zmax.npz, ~1 min
    python diag_jz_to_zmax.py 80 1500    # the validation above
