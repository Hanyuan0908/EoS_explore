"""Diagnostic: the chemical spread of the disc at ONE moment and ONE radius.

Everything else in this project measures abundance scatter across populations
that formed over hundreds of Myr and over a wide range of radius, so a radial
gradient and an age gradient are folded into every dispersion.  This removes
both: take a single snapshot, keep only the stars formed in the DT before it,
and split them by birth radius.  What is left is the scatter of the star-forming
ISM at one place and one time -- the quantity a chemical-evolution model would
call the intrinsic dispersion.

    usage:  diag_newborn_abund_radius.py [T_TARGET] [DT_MYR] [EPS_MIN]
            (default 4.4 50 0.9)

EPS_MIN keeps only stars on near-circular orbits, eps = L_z/L_circ(E) > EPS_MIN,
so the sample is the settled disc and not material a merger is stirring.  eps
comes from birth_orbits_actions.npz, where it is measured in the first stored
snapshot at or after each star formed -- which for a DT window lying inside one
snapshot interval IS this snapshot, in its own potential and disc frame.  The
script checks the window does lie inside one interval and refuses otherwise.

Because eps is only tabulated for the in-situ catalogue, EPS_MIN > 0 also imposes
the in-situ selection; with EPS_MIN = 0 no such cut is made.

T_TARGET is matched to the nearest stored snapshot; the default 4.4 Gyr is 1 Gyr
after the disc spin-up at 3.4 Gyr and ~0.6 Gyr before the GS/E pericentre, so it
samples the quiescent disc rather than the merger.

Stars are "newly formed" if t_snap - DT < t_form <= t_snap, with t_form from
GFM_StellarFormationTime through config_au18.a_to_age.  They are taken in the
snapshot's own disc frame (util.align_galaxy, disc plane on components 1,2) and
within |z| < ZDISC so the sample is the disc and not the halo.

Abundances are per star from GFM_Metals on the Asplund solar scale.  Dispersions
are robust, 1.4826 x MAD, with a bootstrap 16-84 on the dispersion itself.

READ BEFORE USING THE NUMBERS.  Auriga's abundances are not the Milky Way's: the
[Mg/Fe] zero point sits near -0.3 rather than 0, and the dynamic range of every
alpha element is a few hundredths of a dex against the ~0.3 dex separating the
MW's two sequences (FINDINGS.md section 6).  These dispersions describe ISM
inhomogeneity inside the model and must not be set beside an observed sigma.

Writes figures/au18_newborn_abund_t<T>.png
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
DT = (float(sys.argv[2]) if len(sys.argv) > 2 else 50.) / 1000.   # Gyr
EPS_MIN = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
ZDISC = 2.0
ELS = ['C', 'N', 'O', 'Ne', 'Mg', 'Si']
RBINS = [(0., 2.), (2., 4.), (4., 6.), (6., 8.), (8., 12.)]
NMIN, NBOOT = 50, 400
RNG = np.random.default_rng(0)
COLS = ['#5E35B1', '#1F6FB2', '#00897B', '#C8A415', '#E8112D']

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


def sig_err(x):
    if len(x) < NMIN:
        return np.nan, np.nan, np.nan
    b = np.array([mad_sigma(x[RNG.integers(0, len(x), len(x))]) for _ in range(NBOOT)])
    return mad_sigma(x), *np.percentile(b, (16, 84))


st = np.load(C.OUT_DIR + '/snapshot_times.npz')
k = int(np.argmin(np.abs(st['t_snap'] - T_TARGET)))
SNAP, TSNAP = int(st['snaps'][k]), float(st['t_snap'][k])
print(f'target t = {T_TARGET:.2f} Gyr -> snapshot {SNAP} at t = {TSNAP:.3f} Gyr')
print(f'newly formed = {TSNAP - DT:.3f} < t_form <= {TSNAP:.3f} Gyr '
      f'({DT * 1000:.0f} Myr)')
TPREV = float(st['t_snap'][k - 1])
if EPS_MIN > 0 and TSNAP - DT < TPREV:
    raise SystemExit(f'the {DT * 1000:.0f} Myr window reaches below the previous '
                     f'snapshot at t = {TPREV:.3f} Gyr, so eps_birth for some of '
                     f'these stars was measured in a different snapshot; shorten '
                     f'DT or drop the eps cut')

s = snap_mod.load_snapshot(SNAP, 4, snappath=C.SIM_DIR, verbose=False,
    loadlist=['Coordinates', 'Velocities', 'Masses', 'GFM_Metals',
              'GFM_StellarFormationTime', 'ParticleIDs'])
real = s.data['GFM_StellarFormationTime'] > 0
for key in list(s.data):
    s.data[key] = s.data[key][real]
sf = sub_mod.subfind(SNAP, directory=C.SIM_DIR,
                     loadlist=['GroupFirstSub', 'SubhaloPos'])
cen = sf.data['SubhaloPos'][int(sf.data['GroupFirstSub'][0])]
util.CentreOnHalo(s, cen)
util.align_galaxy(s, radialcut=.01)
x = s.data['Coordinates'] * 1e3
# align_galaxy puts the disc axis on component 0; the disc plane is (1, 2)
R = np.hypot(x[:, 1], x[:, 2])
Z = np.abs(x[:, 0])
tform = C.a_to_age(s.data['GFM_StellarFormationTime'])
met = s.data['GFM_Metals']
sid = s.data['ParticleIDs']
del s, x; gc.collect()

new = (tform > TSNAP - DT) & (tform <= TSNAP) & (Z < ZDISC)
if EPS_MIN > 0:
    a = np.load(C.OUT_DIR + '/birth_orbits_actions.npz')
    o = np.argsort(a['ids']); ai = a['ids'][o]
    p = np.searchsorted(ai, sid)
    h = (p < len(ai)) & (ai[np.minimum(p, len(ai) - 1)] == sid)
    eps = np.full(len(sid), np.nan)
    eps[h] = a['eps_birth'][o[p[h]]]
    n0 = int(new.sum())
    new = new & np.isfinite(eps) & (eps > EPS_MIN)
    print(f'eps > {EPS_MIN:g} (and in-situ): {new.sum():,} of {n0:,} '
          f'({100 * new.sum() / max(n0, 1):.1f} %)')
    del a; gc.collect()
feh = C.bracket_abundance(met, 'Fe', 'H')
XFE = {e: C.bracket_abundance(met, e, 'Fe') for e in ELS}
del met; gc.collect()
print(f'newly formed stars within |z| < {ZDISC:g} kpc: {new.sum():,}')

print(f'\n{"R [kpc]":>9s} {"N":>7s} {"med [Fe/H]":>11s} {"sig[Fe/H]":>10s} | '
      + ' '.join(f'{"sig[" + e + "/Fe]":>11s}' for e in ELS))
rows = []
for (lo, hi), col in zip(RBINS, COLS):
    m = new & (R >= lo) & (R < hi) & np.isfinite(feh)
    if m.sum() < NMIN:
        print(f'{f"{lo:g}-{hi:g}":>9s} {m.sum():>7,}   (below N = {NMIN}, skipped)')
        rows.append(None)
        continue
    sf_, slo, shi = sig_err(feh[m])
    line = (f'{f"{lo:g}-{hi:g}":>9s} {m.sum():>7,} {np.median(feh[m]):>+11.3f} '
            f'{sf_:>10.4f} | ')
    sig = {}
    for e in ELS:
        v = XFE[e][m]
        v = v[np.isfinite(v)]
        sig[e] = sig_err(v)
        line += f'{sig[e][0]:>11.4f} '
    print(line)
    rows.append(dict(lo=lo, hi=hi, col=col, m=m, n=int(m.sum()),
                     feh=np.median(feh[m]), sfeh=(sf_, slo, shi), sig=sig))

# ---------------------------------- figure -----------------------------------
fig, AX = plt.subplots(2, 3, figsize=(16.4, 9.2))
for ax, e in zip(AX.flat, ELS):
    for r in rows:
        if r is None:
            continue
        m = r['m'] & np.isfinite(XFE[e])
        ax.scatter(feh[m], XFE[e][m], s=3.5, c=r['col'], alpha=.22, lw=0,
                   rasterized=True)
    for r in rows:
        if r is None:
            continue
        m = r['m'] & np.isfinite(XFE[e])
        ax.plot(np.median(feh[m]), np.median(XFE[e][m]), 'o', color=r['col'],
                ms=10, mec='w', mew=1.4, zorder=5,
                label=f'$R$ = {r["lo"]:g}$-${r["hi"]:g} kpc  '
                      f'($\\sigma$ = {r["sig"][e][0]:.3f})')
    ax.set(xlabel='[Fe/H]', ylabel=f'[{e}/Fe]')
    ax.legend(loc='best', fontsize=10.5, handletextpad=.3, borderpad=.3,
              labelspacing=.3)
sub_ = (rf', $\epsilon > {EPS_MIN:g}$' if EPS_MIN > 0 else '')
fig.suptitle(f'Stars formed in the {DT * 1000:.0f} Myr before $t$ = {TSNAP:.2f} Gyr '
             f'(snapshot {SNAP}), $|z| < {ZDISC:g}$ kpc' + sub_, fontsize=15, y=.995)
fig.tight_layout(pad=.6, w_pad=1.2, h_pad=1.1, rect=[0, 0, 1, .985])
f = (f'{OUT}/au18_newborn_abund_t{TSNAP:.2f}'
     + (f'_eps{EPS_MIN:g}' if EPS_MIN > 0 else '') + '.png')
fig.savefig(f, bbox_inches='tight')
print(f'\nsaved {f}')
