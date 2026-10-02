"""Spherical-approximation orbit quantities, shared by the Auriga and gastro analyses.

Both simulations store a per-particle potential, so the cheapest route to the
quantities the observational analysis uses (r_apo, eccentricity) is to
spherically average that potential and read the turning points off the effective
potential.  For each star, E = 0.5 v^2 + Phi(r) and L = |r x v| give

    Phi_eff(r) = Phi(r) + L^2 / 2 r^2 ,

whose two roots either side of the present radius are r_peri and r_apo.  Both
galaxies have discs, so this is an approximation; it is adequate for the
halo-orbit populations this project cares about and it is what makes ecc and
r_apo comparable between the sims and the APOGEE orbit integrations.
"""
import numpy as np


def potential_profile(r, phi, rmin=0.05, rmax=800., nbin=400, min_count=20):
    """Spherically-averaged Phi(r): bin centres, mean Phi, and the outer Kepler constant."""
    grid = np.logspace(np.log10(rmin), np.log10(rmax), nbin)
    ok = np.isfinite(phi) & (r > 0)
    ib = np.clip(np.searchsorted(grid, r[ok]) - 1, 0, nbin - 2)
    cnt = np.bincount(ib, minlength=nbin - 1)
    tot = np.bincount(ib, weights=phi[ok], minlength=nbin - 1)
    prof = np.full(nbin - 1, np.nan)
    good = cnt > min_count
    prof[good] = tot[good] / cnt[good]
    rc = np.sqrt(grid[:-1] * grid[1:])
    g = np.isfinite(prof)
    rc, prof = rc[g], prof[g]
    return rc, prof, -prof[-1] * rc[-1]


def phi_interp(rq, rc, prof, k_out):
    """Phi at arbitrary radii: interpolated inside the profile, Keplerian outside."""
    rq = np.asarray(rq, float)
    out = np.interp(rq, rc, prof, left=prof[0], right=np.nan)
    far = rq > rc[-1]
    if np.any(far):
        out[far] = -k_out / rq[far]
    return out


def apo_peri(E, L, r_now, rc, prof, k_out, rmin=0.05, rmax=600., ngrid=600, chunk=100_000):
    """r_apo, r_peri from the roots of Phi_eff(r) = E bracketing r_now.

    E must be computed in the *same* spherical potential (see spherical_energy),
    otherwise the flattening of the disc leaves a large fraction of stars below
    the effective-potential floor.

    Stars with no forbidden node outside r_now are unbound within the profile and
    get r_apo = NaN; stars with none inside reach the grid floor (radial orbits).
    Stars that still sit at or below the minimum of Phi_eff are circular orbits
    in this approximation and are assigned r_apo = r_peri = r_circ (ecc = 0).
    """
    solve = np.logspace(np.log10(rmin), np.log10(rmax), ngrid)
    phi_grid = phi_interp(solve, rc, prof, k_out)
    idx = np.arange(ngrid)
    n = len(E)
    rap = np.full(n, np.nan)
    rpe = np.full(n, np.nan)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        phi_eff = phi_grid[None, :] + .5 * L[a:b, None] ** 2 / solve[None, :] ** 2
        f = E[a:b, None] - phi_eff
        i_now = np.clip(np.searchsorted(solve, r_now[a:b]) - 1, 0, ngrid - 1)
        rows = np.arange(b - a)
        allowed = f[rows, i_now] > 0
        forb = f <= 0

        up = forb & (idx[None, :] > i_now[:, None])
        has_up = up.any(1)
        i_up = np.where(has_up, up.argmax(1), ngrid - 1)

        dn = forb & (idx[None, :] < i_now[:, None])
        has_dn = dn.any(1)
        i_dn = np.where(has_dn, ngrid - 1 - dn[:, ::-1].argmax(1), 0)

        def root(i_forb, i_all):
            fa, ff = f[rows, i_all], f[rows, i_forb]
            den = fa - ff
            w = np.where(den != 0, fa / np.where(den != 0, den, 1.), 0.)
            return solve[i_all] + w * (solve[i_forb] - solve[i_all])

        r_circ = solve[phi_eff.argmin(1)]
        rap[a:b] = np.where(allowed,
                            np.where(has_up, root(i_up, np.maximum(i_up - 1, 0)), np.nan),
                            r_circ)
        rpe[a:b] = np.where(allowed,
                            np.where(has_dn, root(i_dn, np.minimum(i_dn + 1, ngrid - 1)), solve[0]),
                            r_circ)
    return rap, rpe


def spherical_energy(v2, r, rc, prof, k_out):
    """Orbital energy in the spherically-averaged potential (0.5 v^2 + Phi_sph(r))."""
    return .5 * v2 + phi_interp(r, rc, prof, k_out)


def eccentricity(rap, rpe):
    tot = rap + rpe
    return (rap - rpe) / np.where(tot > 0, tot, np.nan)


def density_contours(ax, x, y, rng, color, label=None, levels=(0.9, 0.6, 0.3), bins=90,
                     smooth=1.2, lw=1.6, ls='-'):
    """Contours enclosing the given fractions of a population, over a 2D map.

    Cleaner than scattering two overlapping populations on top of a density map:
    each level is the contour inside which that fraction of the sample lies.
    """
    from scipy.ndimage import gaussian_filter
    ok = np.isfinite(x) & np.isfinite(y)
    h, xe, ye = np.histogram2d(x[ok], y[ok], bins=bins, range=rng)
    h = gaussian_filter(h, smooth)
    flat = np.sort(h.ravel())[::-1]
    csum = np.cumsum(flat) / max(flat.sum(), 1)
    vals = sorted({float(flat[np.searchsorted(csum, f)]) for f in levels
                   if np.searchsorted(csum, f) < len(flat)})
    if not vals:
        return
    ax.contour(.5 * (xe[:-1] + xe[1:]), .5 * (ye[:-1] + ye[1:]), h.T, levels=vals,
               colors=color, linewidths=lw, linestyles=ls)
    if label:
        ax.plot([], [], color=color, lw=lw, ls=ls, label=label)


def local_enhancement(t, mask, base, window, gap=0.4, width=0.8):
    """How much more likely a birth cohort inside `window` is to satisfy `mask`.

    Compared with the mean of two control intervals of the same width placed
    `gap` before and after the window, so a slow secular trend does not masquerade
    as a merger-induced enhancement.
    """
    def frac(lo, hi):
        s = base & (t >= lo) & (t < hi)
        return mask[s].mean() if s.sum() > 100 else np.nan

    inside = frac(*window)
    before = frac(window[0] - gap - width, window[0] - gap)
    after = frac(window[1] + gap, window[1] + gap + width)
    control = np.nanmean([before, after])
    return inside, control, (inside / control if control > 0 else np.nan)
