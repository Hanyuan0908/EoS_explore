"""Diagnostic: the abundance plane of one NARROW annulus of star-forming gas.

diag_gas_abund_radius uses 2 kpc annuli, inside which the radial gradient itself
contributes to the scatter.  This takes a single thin ring so that what is left
is as close as this simulation gets to the spread at one radius, and then asks
where that residual spread comes from.

    usage:  diag_gas_abund_annulus.py [T_TARGET] [R_LO] [R_HI]
            (default 4.4 4.0 4.5)

Four decompositions of sigma[Fe/H] are printed, each removing one candidate
cause:

  raw            all cells in the ring
  de-trended     after subtracting a linear fit in R across the ring, so the
                 residual radial gradient cannot contribute
  per azimuth    the median sigma within 30 degree sectors -- if the spread is
                 spiral structure, a sector is much tighter than the ring
  per |z| slice  the same for three height slices, since a thick ring mixes
                 midplane and off-plane gas

Panel (h) is the ring seen face-on and coloured by [Fe/H], which is the direct
way to see whether the spread is azimuthal structure or local scatter.

Same frame convention as diag_gas_abund_radius: the rotation comes from the STARS
and is applied to the gas by hand, never by calling align_galaxy on the gas.

Writes figures/au18_gas_annulus_t<T>_R<LO>-<HI>.png
"""
import gc, os, sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod, util

OUT = C.FIG_DIR
os.makedirs(OUT, exist_ok=True)
T_TARGET = float(sys.argv[1]) if len(sys.argv) > 1 else 4.4
R_LO = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
R_HI = float(sys.argv[3]) if len(sys.argv) > 3 else 4.5
ZDISC = 2.0
ELS = ['C', 'N', 'O', 'Ne', 'Mg', 'Si']
NMIN = 50

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 14, 'axes.labelsize': 15.5,
    'xtick.labelsize': 13.5, 'ytick.labelsize': 13.5, 'legend.fontsize': 12,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 140, 'savefig.dpi': 200,
})


def msig(x):
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


st = np.load(C.OUT_DIR + '/snapshot_times.npz')
k = int(np.argmin(np.abs(st['t_snap'] - T_TARGET)))
SNAP, TSNAP = int(st['snaps'][k]), float(st['t_snap'][k])
print(f'snapshot {SNAP}, t = {TSNAP:.3f} Gyr; ring {R_LO:g}-{R_HI:g} kpc')

sub = sub_mod.subfind(SNAP, directory=C.SIM_DIR,
                      loadlist=['SubhaloPos', 'Group_R_Crit200'])
r200 = float(sub.data['Group_R_Crit200'][0]); cen = sub.data['SubhaloPos'][0]
ref = snap_mod.load_snapshot(SNAP, 4, snappath=C.SIM_DIR, verbose=False,
    loadlist=['Coordinates', 'Masses', 'Potential', 'Velocities'])
ref = util.CentreOnHalo(ref, cen)
ref = util.apply_mask(ref, stars=False, radialcut=.5 * r200)
ist, = np.where(util.r(ref) < .1 * r200)
L = np.cross(ref.data['Coordinates'][ist],
             ref.data['Velocities'][ist] * ref.data['Masses'][ist, None])
Ld = L.sum(0); Ld /= np.sqrt((Ld ** 2).sum())
xd, yd, zd = util.get_principal_axis(ref, ist, L=Ld)
del ref; gc.collect()

g = snap_mod.load_snapshot(SNAP, 0, snappath=C.SIM_DIR, verbose=False,
    loadlist=['Coordinates', 'Masses', 'GFM_Metals', 'StarFormationRate'])
g = util.CentreOnHalo(g, cen)
util.rotateto(g, xd, dir2=yd, dir3=zd)
x = g.data['Coordinates'] * 1e3
X, Y, Z = x[:, 1], x[:, 2], x[:, 0]
R = np.hypot(X, Y)
sfr = np.asarray(g.data['StarFormationRate'], float)
met = g.data['GFM_Metals']
del g, x; gc.collect()

feh = C.bracket_abundance(met, 'Fe', 'H')
XFE = {e: C.bracket_abundance(met, e, 'Fe') for e in ELS}
del met; gc.collect()
m = (R >= R_LO) & (R < R_HI) & (np.abs(Z) < ZDISC) & (sfr > 0) & np.isfinite(feh)
print(f'star-forming cells in the ring, |z| < {ZDISC:g} kpc: {m.sum():,}')
Rr, Xr, Yr, Zr, F = R[m], X[m], Y[m], Z[m], feh[m]
PHI = np.degrees(np.arctan2(Yr, Xr)) % 360.

print(f'\nmedian [Fe/H] = {np.median(F):+.3f}')
print(f'  raw sigma[Fe/H]                        {msig(F):.4f}')
sl = np.polyfit(Rr, F, 1)[0]
print(f'  after removing the radial trend         {msig(F - np.polyval(np.polyfit(Rr, F, 1), Rr)):.4f}'
      f'   (gradient {sl:+.3f} dex/kpc, i.e. {abs(sl) * (R_HI - R_LO):.3f} dex across the ring)')
sec = [msig(F[(PHI >= a) & (PHI < a + 30)]) for a in range(0, 360, 30)]
print(f'  median within 30 deg sectors            {np.nanmedian(sec):.4f}'
      f'   (sector range {np.nanmin(sec):.4f}-{np.nanmax(sec):.4f})')
zs = [msig(F[np.abs(Zr) < .25]), msig(F[(np.abs(Zr) >= .25) & (np.abs(Zr) < 1.)]),
      msig(F[np.abs(Zr) >= 1.])]
print(f'  within |z| < 0.25 / 0.25-1 / > 1 kpc    '
      + ' / '.join(f'{v:.4f}' for v in zs))
print(f'\n{"element":>8s} {"median":>9s} {"sigma":>8s} {"de-trended":>11s}')
for e in ELS:
    v = XFE[e][m]
    dv = v - np.polyval(np.polyfit(Rr[np.isfinite(v)], v[np.isfinite(v)], 1), Rr)
    print(f'{e:>8s} {np.nanmedian(v):>+9.4f} {msig(v):>8.4f} {msig(dv):>11.4f}')

# ---------------------------------- figure -----------------------------------
fig, AX = plt.subplots(2, 4, figsize=(21.0, 9.2))
# explicit slots: AX.flat would put the sixth element in [0,3], which the
# face-on map then overwrites, silently losing a panel
EAX = [AX[0, 0], AX[0, 1], AX[0, 2], AX[1, 0], AX[1, 1], AX[1, 2]]
FR = (np.percentile(F, .2), np.percentile(F, 99.8))
for ax, e in zip(EAX, ELS):
    v = XFE[e][m]
    ok = np.isfinite(v)
    yr = (np.percentile(v[ok], .2), np.percentile(v[ok], 99.8))
    h, xe, ye = np.histogram2d(F[ok], v[ok], bins=(110, 110), range=[FR, yr])
    ax.pcolormesh(xe, ye, np.where(h > 0, h, np.nan).T, cmap='magma_r',
                  norm=LogNorm(vmin=1, vmax=h.max()), rasterized=True)
    # the running median, to show the shape rather than assert it
    eb = np.linspace(*FR, 26)
    ct = .5 * (eb[:-1] + eb[1:])
    md = [np.median(v[ok & (F >= a) & (F < b)]) if (ok & (F >= a) & (F < b)).sum() > 30
          else np.nan for a, b in zip(eb[:-1], eb[1:])]
    ax.plot(ct, md, color='#00E5FF', lw=2.2)
    ax.set(xlabel='[Fe/H]', ylabel=f'[{e}/Fe]', xlim=FR, ylim=yr)
    ax.text(.04, .95, f'$\\sigma$ = {msig(v):.4f}', transform=ax.transAxes,
            va='top', fontsize=13,
            bbox=dict(fc='white', ec='none', alpha=.85, pad=2))

ax = AX[1, 3]
ax.hist(F, bins=60, color='#5E35B1', histtype='stepfilled', alpha=.35, lw=0)
ax.hist(F, bins=60, color='#5E35B1', histtype='step', lw=2.2)
ax.axvline(np.median(F), color='k', lw=1.4, ls='--')
ax.set(xlabel='[Fe/H]', ylabel='cells per bin', xlim=FR)
ax.text(.04, .95, f'$\\sigma$ = {msig(F):.4f}\nmedian = {np.median(F):+.3f}',
        transform=ax.transAxes, va='top', fontsize=13,
        bbox=dict(fc='white', ec='none', alpha=.85, pad=2))

ax = AX[0, 3]
sc = ax.scatter(Xr, Yr, s=4.5, c=F, cmap='viridis', vmin=FR[0], vmax=FR[1],
                lw=0, alpha=.8, rasterized=True)
cb = fig.colorbar(sc, ax=ax, fraction=.046, pad=.015)
cb.set_label('[Fe/H]')
ax.set(aspect='equal', xlabel='$x$ [kpc]', ylabel='$y$ [kpc]',
       xlim=(-R_HI * 1.1, R_HI * 1.1), ylim=(-R_HI * 1.1, R_HI * 1.1))
ax.text(.04, .95, 'the ring, face-on', transform=ax.transAxes, va='top',
        fontsize=13, bbox=dict(fc='white', ec='none', alpha=.85, pad=2))

fig.suptitle(f'Star-forming gas, ${R_LO:g} < R < {R_HI:g}$ kpc, $|z| < {ZDISC:g}$ kpc, '
             f'at $t$ = {TSNAP:.2f} Gyr (snapshot {SNAP}); {m.sum():,} cells',
             fontsize=15, y=.985)
fig.tight_layout(pad=.6, w_pad=1.3, h_pad=1.1, rect=[0, 0, 1, .975])
f = f'{OUT}/au18_gas_annulus_t{TSNAP:.2f}_R{R_LO:g}-{R_HI:g}.png'
fig.savefig(f, bbox_inches='tight')
print(f'\nsaved {f}')
