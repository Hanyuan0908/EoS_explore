"""Diagnostic: the [X/Fe]-[Fe/H] plane of the GAS, coloured by radius.

The companion to diag_newborn_abund_radius, which does the same for the stars
formed in the 50 Myr before a snapshot.  The gas is the more direct measurement:
it is the material itself rather than a sample of it drawn by star formation, so
it carries no selection by where star formation happened to be going on, and no
dependence on how long a window is taken.

    usage:  diag_gas_abund_radius.py [T_TARGET] [R_MAX] [MODE]
            (default 4.4 8 sf)

MODE 'sf'  star-forming cells only (StarFormationRate > 0), the same definition
           prep_gas_disc_au18.py uses: Auriga's own flag, n_H > 0.13 cm^-3.  This
           is the material that will actually become stars, so it is what the
           newborn-star figure should be read against.
MODE 'all' every cell in the aperture, star-forming or not.

R_MAX = 8 kpc by default so the colour scale spends its whole range on the inner
disc; the radial gradient is steep and a 0-30 kpc scale compresses everything of
interest into one end of the bar.

FRAME.  The rotation is taken from the STARS inside 10 kpc and applied by hand to
the gas, so both components share one frame -- exactly as prep_gas_disc_au18.py
does.  Calling util.align_galaxy on the gas object instead would align to the
GAS angular momentum, which is not the same axis and would silently put the two
components in different frames.  The disc plane is components (1, 2).

DISPERSIONS ARE UNWEIGHTED.  Arepo keeps cell masses near a target value, so
mass-weighting changes little; the script prints the 16-84 spread of cell mass so
that can be checked rather than assumed.

The usual caveat: Auriga's [Mg/Fe] zero point sits near -0.3 rather than 0 and
the dynamic range of every alpha element is a few hundredths of a dex, against
the ~0.3 dex separating the Milky Way's two sequences (FINDINGS.md section 6).
These are ISM inhomogeneities inside the model.

Writes figures/au18_gas_abund_t<T>_<MODE>.png
"""
import gc, os, sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod, util

OUT = C.FIG_DIR
os.makedirs(OUT, exist_ok=True)
T_TARGET = float(sys.argv[1]) if len(sys.argv) > 1 else 4.4
R_MAX = float(sys.argv[2]) if len(sys.argv) > 2 else 8.
MODE = sys.argv[3] if len(sys.argv) > 3 else 'sf'
assert MODE in ('sf', 'all')
ZDISC = 2.0
ELS = ['C', 'N', 'O', 'Ne', 'Mg', 'Si']
RBINS = [(0., 2.), (2., 4.), (4., 6.), (6., 8.)]
NMIN, NBOOT = 50, 400
RNG = np.random.default_rng(0)

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 14, 'axes.labelsize': 15.5,
    'xtick.labelsize': 13.5, 'ytick.labelsize': 13.5, 'legend.fontsize': 12.5,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 140, 'savefig.dpi': 200,
})


def mad_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


st = np.load(C.OUT_DIR + '/snapshot_times.npz')
k = int(np.argmin(np.abs(st['t_snap'] - T_TARGET)))
SNAP, TSNAP = int(st['snaps'][k]), float(st['t_snap'][k])
print(f'target t = {T_TARGET:.2f} Gyr -> snapshot {SNAP} at t = {TSNAP:.3f} Gyr; '
      f'mode = {MODE}')

# the frame comes from the STARS and is applied by hand to the gas
sub = sub_mod.subfind(SNAP, directory=C.SIM_DIR,
                      loadlist=['SubhaloPos', 'Group_R_Crit200'])
r200 = float(sub.data['Group_R_Crit200'][0])
cen = sub.data['SubhaloPos'][0]
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
R = np.hypot(x[:, 1], x[:, 2])
Z = np.abs(x[:, 0])
gm = g.data['Masses'] * C.MASS_TO_MSUN
sfr = np.asarray(g.data['StarFormationRate'], float)
met = g.data['GFM_Metals']
del g, x; gc.collect()

sel = (R < R_MAX) & (Z < ZDISC)
if MODE == 'sf':
    sel = sel & (sfr > 0)
feh = C.bracket_abundance(met, 'Fe', 'H')
XFE = {e: C.bracket_abundance(met, e, 'Fe') for e in ELS}
del met; gc.collect()
sel = sel & np.isfinite(feh)
print(f'cells with R < {R_MAX:g} kpc, |z| < {ZDISC:g} kpc'
      + (' and SFR > 0' if MODE == 'sf' else '') + f': {sel.sum():,}')
q = np.percentile(gm[sel], [16, 50, 84])
print(f'cell mass 16/50/84 = {q[0]:.3e} / {q[1]:.3e} / {q[2]:.3e} Msun '
      f'(spread {q[2] / q[0]:.2f}x, so weighting matters little)')
print(f'total SFR in the aperture: {sfr[sel].sum():.2f} Msun/yr')

print(f'\n{"R [kpc]":>9s} {"N":>8s} {"med [Fe/H]":>11s} {"sig[Fe/H]":>10s} | '
      + ' '.join(f'{"sig[" + e + "/Fe]":>11s}' for e in ELS))
for lo, hi in RBINS:
    m = sel & (R >= lo) & (R < hi)
    if m.sum() < NMIN:
        print(f'{f"{lo:g}-{hi:g}":>9s} {m.sum():>8,}   (below N = {NMIN})')
        continue
    line = (f'{f"{lo:g}-{hi:g}":>9s} {m.sum():>8,} {np.median(feh[m]):>+11.3f} '
            f'{mad_sigma(feh[m]):>10.4f} | ')
    for e in ELS:
        v = XFE[e][m]
        line += f'{mad_sigma(v[np.isfinite(v)]):>11.4f} '
    print(line)

# ---------------------------------- figure -----------------------------------
i = np.flatnonzero(sel)
RNG.shuffle(i)                 # plot order must not bias which radius sits on top
fig, AX = plt.subplots(2, 3, figsize=(16.6, 9.2))
for ax, e in zip(AX.flat, ELS):
    j = i[np.isfinite(XFE[e][i])]
    sc = ax.scatter(feh[j], XFE[e][j], s=3.0, c=R[j], cmap='plasma', vmin=0,
                    vmax=R_MAX, alpha=.35, lw=0, rasterized=True)
    for lo, hi in RBINS:
        m = sel & (R >= lo) & (R < hi) & np.isfinite(XFE[e])
        if m.sum() < NMIN:
            continue
        ax.plot(np.median(feh[m]), np.median(XFE[e][m]), 'o',
                color=plt.get_cmap('plasma')(.5 * (lo + hi) / R_MAX), ms=11,
                mec='k', mew=1.3, zorder=5)
    ax.set(xlabel='[Fe/H]', ylabel=f'[{e}/Fe]')
cb = fig.colorbar(sc, ax=AX, fraction=.018, pad=.012)
cb.solids.set_alpha(1.)
cb.set_label('$R$ [kpc]')
lab = 'star-forming gas' if MODE == 'sf' else 'all gas'
fig.suptitle(f'{lab} at $t$ = {TSNAP:.2f} Gyr (snapshot {SNAP}), '
             f'$R < {R_MAX:g}$ kpc, $|z| < {ZDISC:g}$ kpc; '
             'rings mark the median of each 2 kpc annulus', fontsize=15, y=.98)
f = f'{OUT}/au18_gas_abund_t{TSNAP:.2f}_{MODE}.png'
fig.savefig(f, bbox_inches='tight')
print(f'\nsaved {f}')
