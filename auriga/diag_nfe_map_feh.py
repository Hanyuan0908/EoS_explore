"""Verification: are the two [N/Fe] components spatially separated, bin by bin?

The claim from diag_nfe_bimodality is that the bimodal [N/Fe] distribution of the
bridge is two unmixed gas streams -- GS/E gas at the satellite end of the lane,
host-disc gas at the host end.  If that is right, then in any [Fe/H] bin where
BOTH components exist, a map of the halo-born stars coloured by [N/Fe] must show
a gradient along the lane, and the [N/Fe] of a star must correlate with how far
it is from the GS/E centroid.  If the bimodality were instead an unresolved
mixture with no spatial meaning, the colours would be scrambled.

One panel per 0.2 dex bin of [Fe/H], all halo-born stars born in the snapshot-72
window, x-z in the same frame as au18_birth_positions_gas4.  The colour scale is
shared across panels and diverging about [N/Fe] = 0.09, the branch divider
measured in diag_nfe_dispersion_origin -- so blue IS the N-poor (GS/E-like)
component and red IS the N-rich (host-like) one, and the pale band is the divide
between them rather than an arbitrary midpoint.

cmasher's iceburn was tried here and rejected: its midpoint is black, and because
most of this sample sits near the divider, whole panels fell into the dark zone
(the [Fe/H] = -0.8 to -0.6 panel, median [N/Fe] = +0.063, went uniformly black and
lost its gradient).  A pale-centred diverging map keeps every panel legible.

Marker OPACITY is scaled as 900/N per panel (clipped to 0.20-0.95), not marker
size.  N runs from 112 to 3,777 across the bins, and at one fixed opacity the
crowded intermediate bins saturate into a solid block that hides the interface
between the two streams.  Letting overlapping markers build up instead keeps the
density contrast that a uniform-opacity scatter throws away -- the same device
au18_birth_positions_gas4 uses, and for the same reason.

Each panel carries the Spearman rho between distance from the GS/E centroid and
[N/Fe], which is the claim stated as a number: rho > 0 means the further from the
satellite, the more N-rich.

Writes figures/au18_nfe_map_feh.png
"""
import gc, os, sys
import numpy as np
from scipy.stats import spearmanr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import auriga_public as ap
import config_au18 as C
import au18_frame as AF

SNAP = 72
CUT, ZCUT = 0.8, 1.5
# stops at +0.4: the +0.4 to +0.6 bin holds 22 halo-born stars, too few to
# read a gradient from, and it would force a third row of empty panels
FBINS = np.arange(-1.2, 0.4 + 1e-9, 0.2)
NMIN = 40
R_SAT, R_HOST = 6.0, 8.0
DIVIDER = 0.09                 # the measured branch divider near [Fe/H] ~ -0.3
VLO, VHI = 0.00, 0.19
cSAT, cHOST = '#8E24AA', '#B8860B'
OUT = os.path.dirname(os.path.abspath(__file__)) + '/figures'
RNG = np.random.default_rng(0)

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 13, 'axes.labelsize': 14.5,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 140, 'savefig.dpi': 200,
})

# ------------------------------- load ----------------------------------------
a = np.load(C.OUT_DIR + '/birth_orbits_actions.npz')
zx = np.load(C.OUT_DIR + '/birth_orbits_zmax.npz')
st = np.load(C.OUT_DIR + '/snapshot_times.npz')
GSE = np.sort(np.load(C.OUT_DIR + '/gse_clean_ids.npy'))
SN_ALL, T_ALL = st['snaps'], st['t_snap']
k = int(np.flatnonzero(SN_ALL == SNAP)[0])
t_lo, t_hi = T_ALL[k - 1], T_ALL[k]

sub = ap.subhalos.subfind(SNAP, directory=C.SIM_DIR,
                          loadlist=['SubhaloPos', 'Group_R_Crit200'])
r200 = float(sub.data['Group_R_Crit200'][0]); cen = sub.data['SubhaloPos'][0]
ref = ap.snapshot.load_snapshot(SNAP, 4, snappath=C.SIM_DIR, verbose=False,
    loadlist=['Coordinates', 'Masses', 'Potential', 'Velocities'])
ref = ap.util.CentreOnHalo(ref, cen)
ref = ap.util.apply_mask(ref, stars=False, radialcut=.5 * r200)
ist, = np.where(ap.util.r(ref) < .1 * r200)
L = np.cross(ref.data['Coordinates'][ist],
             ref.data['Velocities'][ist] * ref.data['Masses'][ist, None])
Ld = L.sum(0); Ld /= np.sqrt((Ld ** 2).sum())
xd, yd, zd = ap.util.get_principal_axis(ref, ist, L=Ld)
del ref; gc.collect()

s = ap.snapshot.load_snapshot(SNAP, 4, snappath=C.SIM_DIR, verbose=False,
    loadlist=['Coordinates', 'ParticleIDs', 'GFM_Metals'])
s = ap.util.CentreOnHalo(s, cen)
ap.util.rotateto(s, xd, dir2=yd, dir3=zd)
cc = s.data['Coordinates']
pos = np.column_stack([cc[:, 1], cc[:, 2], cc[:, 0]]) * 1e3
sids, met = s.data['ParticleIDs'], s.data['GFM_Metals']
del s; gc.collect()

gcen, _, _, _ = AF.gse_centroid(pos, sids, GSE)
ROT = AF.align_azimuth(gcen)
pos = pos @ ROT.T
g2 = ROT @ gcen

sel = (a['tform'] > t_lo) & (a['tform'] <= t_hi) & np.isfinite(a['eps_birth'])
ids_w, eps_w, zmx_w = a['ids'][sel], a['eps_birth'][sel], zx['zmax_birth'][sel]
o = np.argsort(sids); ss = sids[o]
p = np.searchsorted(ss, ids_w)
ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == ids_w)
P, M = pos[o[p[ok]]], met[o[p[ok]]]
eps, zmx = eps_w[ok], zmx_w[ok]
disc = (eps > CUT) | (zmx < ZCUT)
del met, pos; gc.collect()

feh = C.bracket_abundance(M, 'Fe', 'H')
nfe = C.bracket_abundance(M, 'N', 'Fe')
halo = (~disc) & np.isfinite(feh) & np.isfinite(nfe)
d_gse = np.linalg.norm(P - g2, axis=1)
d_host = np.linalg.norm(P, axis=1)

print(f'snap {SNAP}, t_form {t_lo:.2f}-{t_hi:.2f} Gyr; halo-born = {halo.sum():,}')
print(f'GS/E centroid at (x,z) = ({g2[0]:.1f},{g2[2]:.1f}) kpc\n')
print(f'{"[Fe/H] bin":>16s} {"N":>6s} {"med [N/Fe]":>11s} '
      f'{"rho(d_GSE,[N/Fe])":>19s} {"p":>10s} {"alpha":>7s}')

# ---------------------------------- figure -----------------------------------
NC = 4
NR = int(np.ceil((len(FBINS) - 1) / NC))
fig, AX = plt.subplots(NR, NC, figsize=(4.35 * NC, 4.6 * NR),
                       sharex=True, sharey=True)
norm = TwoSlopeNorm(vmin=VLO, vcenter=DIVIDER, vmax=VHI)
th = np.linspace(0, 2 * np.pi, 200)
sc = None
ALPHA = {}

for i in range(len(FBINS) - 1):
    ax = AX.flat[i]
    b = halo & (feh >= FBINS[i]) & (feh < FBINS[i + 1])
    n = int(b.sum())
    # every window star, faint, so the lane is visible even in sparse bins
    ax.scatter(P[halo, 0], P[halo, 2], s=1.6, c='0.88', lw=0, rasterized=True)
    if n:
        j = np.flatnonzero(b)
        RNG.shuffle(j)                 # plot order must not bias which colour is on top
        # opacity scaled to the panel's own N, so crowded bins stay readable
        al = float(np.clip(900. / n, .20, .95))
        ALPHA[i] = al
        sc = ax.scatter(P[j, 0], P[j, 2], s=13, c=nfe[j], cmap='RdYlBu_r',
                        norm=norm, lw=0, alpha=al, rasterized=True)
    ax.plot(R_SAT * np.cos(th) + g2[0], R_SAT * np.sin(th) + g2[2], color=cSAT, lw=1.5)
    ax.plot(R_HOST * np.cos(th), R_HOST * np.sin(th), color=cHOST, lw=1.5)
    ax.set(aspect='equal', xlim=(-25, 25), ylim=(-25, 25))
    lab = f'[Fe/H] = {FBINS[i]:+.1f} to {FBINS[i+1]:+.1f}'
    bb = dict(fc='white', ec='none', alpha=.8, pad=1.8)
    ax.text(.035, .965, lab, transform=ax.transAxes, va='top', fontsize=12, bbox=bb)
    ax.text(.035, .888, f'N = {n:,}', transform=ax.transAxes, va='top',
            fontsize=11.5, bbox=bb)
    if n >= NMIN:
        rho, pv = spearmanr(d_gse[b], nfe[b])
        ax.text(.035, .045, r'$\rho$ = ' + f'{rho:+.2f}', transform=ax.transAxes,
                fontsize=12.5, fontweight='bold', bbox=bb)
        print(f'{lab:>16s} {n:>6,} {np.median(nfe[b]):>+11.3f} {rho:>19.3f} '
              f'{pv:>10.1e} {ALPHA.get(i, np.nan):>7.2f}')
    else:
        print(f'{lab:>16s} {n:>6,} {np.median(nfe[b]) if n else np.nan:>+11.3f} '
              f'{"too few":>19s} {"--":>10s}')

for i in range(len(FBINS) - 1, NR * NC):
    AX.flat[i].axis('off')
for ax in AX[-1, :]:
    ax.set_xlabel('$x$ [kpc]')
for ax in AX[:, 0]:
    ax.set_ylabel('$z$ [kpc]')

fig.subplots_adjust(right=.90)
cax = fig.add_axes([.915, .18, .013, .64])
cb = fig.colorbar(sc, cax=cax)
cb.set_label('[N/Fe]', fontsize=14, labelpad=30)
cb.ax.axhline(DIVIDER, color='k', lw=1.4)
cax.text(1.9, DIVIDER, ' branch\n divider', transform=cb.ax.get_yaxis_transform(),
         va='center', fontsize=10)

fig.suptitle('Halo-born stars, snapshot 72, coloured by [N/Fe]; violet = satellite '
             'aperture, gold = host aperture', fontsize=13.5, y=.965)
f = f'{OUT}/au18_nfe_map_feh.png'
fig.savefig(f, bbox_inches='tight')
print('\nsaved', f)
