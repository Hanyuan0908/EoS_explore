"""Diagnostic, NOT a paper figure: the gas maps restricted to STAR-FORMING gas.

au18_gas_metallicity and its per-snapshot companion show ALL gas.  Most of that
is diffuse halo material that will never form a star, so the metallicity map is
dominated by a reservoir irrelevant to the burst.  This repeats the two panels
for the gas that is actually eligible to form stars.

STAR-FORMING GAS = cells with Auriga's own StarFormationRate > 0 flag, the same
definition prep_gas_disc_au18.py uses.  Grand et al. (2017) give the threshold as
a hydrogen number density n_H > 0.13 cm^-3; the minimum density among SF-flagged
cells measured in this run is 0.107 cm^-3, ~20 per cent lower, consistent once
the fixed X_H = 0.76 behind that estimate is allowed for.

NOT a temperature cut, deliberately.  Auriga puts star-forming gas on the
Springel & Hernquist effective equation of state, so its InternalEnergy is a
pressure floor and not a real temperature -- only 66 per cent of SF cells at
z = 0 fall below 3e4 K.  A (T, n) cut from another code cannot be carried over
either: GASTRO and VINTERGATAN both require n > 1 cm^-3, eight times Auriga's
threshold, which here would select only the nucleus.

    usage:  diag_gas_metallicity_sf.py [SNAP]        (default 72)

Frame, azimuth alignment and panel layout are identical to
diag_gas_metallicity_snap.py, so the two can be laid side by side.  The maps are
built here rather than read from gas_chem_maps_*, because that cache accumulated
its moments over all gas and the SF subset cannot be recovered from it.

Two threshold changes against the all-gas version, both because SF gas is dense
by construction:
  * no surface-density blanking (SMIN = 0).  The all-gas map blanks faint pixels
    because the frame is mostly diffuse material; that problem does not arise.
  * MMIN, the mass a pixel needs before its mean [Fe/H] is drawn, is 1e5 Msun
    rather than 5e5 -- a few cells, not a few tens.

Also prints the SF gas mass in the satellite core / envelope / wake, the same
shells as diag_gse_gas_shells.py, so the stripping question can be asked of the
star-forming reservoir specifically.

PROJECTION DEPTH.  Like the all-gas version, the maps are cut in x and z but not
in y, so each pixel integrates through the whole box along the line of sight.
For diffuse gas that only adds a smooth background; for SF gas it could in
principle project an unrelated star-forming galaxy into the frame.  Nothing in
these three snapshots looks like that, but the possibility is why the totals
above are quoted inside 50 kpc of the host rather than read off the map.

Writes figures/au18_gas_metallicity_sf_snap<NN>.png -- figures/, not Fig_paper/.
"""
import gc, os, sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import auriga_public as ap
import config_au18 as C
import au18_frame as AF
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import orbit_tools as OT

OUT = C.FIG_DIR
os.makedirs(OUT, exist_ok=True)
SNAP = int(sys.argv[1]) if len(sys.argv) > 1 else 72
VMIN, VMAX = -0.7, 0.0                  # fixed across snapshots, as in the all-gas version
SIG_LO, SIG_HI = 1e6, 1e9
NB, MMIN = 240, 1e5
R_CORE, R_AP, R_WAKE = 3.0, 6.0, 15.0

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 13.5, 'axes.labelsize': 15,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12.5,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 150, 'savefig.dpi': 200,
})

st_ = np.load(C.OUT_DIR + '/snapshot_times.npz')
TSNAP = float(st_['t_snap'][np.flatnonzero(st_['snaps'] == SNAP)[0]])

# ------------------------------- frame ---------------------------------------
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

s4 = ap.snapshot.load_snapshot(SNAP, 4, snappath=C.SIM_DIR, verbose=False,
    loadlist=['Coordinates', 'ParticleIDs'])
s4 = ap.util.CentreOnHalo(s4, cen); ap.util.rotateto(s4, xd, dir2=yd, dir3=zd)
c4 = s4.data['Coordinates']
sp = np.column_stack([c4[:, 1], c4[:, 2], c4[:, 0]]) * 1e3
sid = s4.data['ParticleIDs']; del s4, c4; gc.collect()
gid = np.sort(np.load(C.OUT_DIR + '/gse_clean_ids.npy'))
o = np.argsort(sid); ss = sid[o]; p = np.searchsorted(ss, gid)
okg = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == gid)
G = sp[o[p[okg]]]; del sp, sid; gc.collect()
ROT = AF.align_azimuth(np.median(G, axis=0))
G = G @ ROT.T
gc_ = np.median(G, axis=0)
XLIM = float(max(30., abs(gc_[0]) + 9.)); ZLIM = float(max(24., abs(gc_[2]) + 7.))

# ------------------------------- the gas -------------------------------------
g = ap.snapshot.load_snapshot(SNAP, 0, snappath=C.SIM_DIR, verbose=False,
    loadlist=['Coordinates', 'Masses', 'StarFormationRate', 'GFM_Metals'])
g = ap.util.CentreOnHalo(g, cen); ap.util.rotateto(g, xd, dir2=yd, dir3=zd)
cg = g.data['Coordinates']
gp = (np.column_stack([cg[:, 1], cg[:, 2], cg[:, 0]]) * 1e3) @ ROT.T
gm = g.data['Masses'] * C.MASS_TO_MSUN
sfr = np.asarray(g.data['StarFormationRate'], float)
feh = C.bracket_abundance(g.data['GFM_Metals'], 'Fe', 'H')
del g, cg; gc.collect()

SF = sfr > 0
# Quote totals inside R_GAL of the host, NOT over the loaded arrays: the snapshot
# holds the whole parent box, so a global sum is ~6e15 Msun of gas and thousands
# of Msun/yr of star formation belonging to unrelated galaxies.
R_GAL = 50.
loc = np.linalg.norm(gp, axis=1) < R_GAL
print(f'snapshot {SNAP}, t = {TSNAP:.3f} Gyr; frame {XLIM:.0f} x {ZLIM:.0f} kpc')
print(f'  within {R_GAL:.0f} kpc of the host: gas {gm[loc].sum():.3e} Msun, '
      f'star-forming {gm[loc & SF].sum():.3e} '
      f'({100 * gm[loc & SF].sum() / gm[loc].sum():.2f} %), '
      f'SFR {sfr[loc].sum():.2f} Msun/yr')


def wmed(x, w):
    if not len(x):
        return np.nan
    i = np.argsort(x); x, w = x[i], w[i]
    c = np.cumsum(w)
    return float(np.interp(.5 * c[-1], c, x))


dg = np.linalg.norm(gp - gc_, axis=1)
fin = np.isfinite(feh)
print(f'  {"shell":<22s} {"M_SF":>10s} {"[Fe/H]_SF":>10s} {"M_all":>10s} {"[Fe/H]_all":>11s}')
for nm, m in (('core      < 3 kpc', dg < R_CORE),
              ('envelope  3-6 kpc', (dg >= R_CORE) & (dg < R_AP)),
              ('wake     6-15 kpc', (dg >= R_AP) & (dg < R_WAKE))):
    a_, b_ = m & fin, m & fin & SF
    print(f'  {nm:<22s} {gm[b_].sum():>10.3e} {wmed(feh[b_], gm[b_]):>+10.3f} '
          f'{gm[a_].sum():>10.3e} {wmed(feh[a_], gm[a_]):>+11.3f}')

ins = SF & (np.abs(gp[:, 0]) < XLIM) & (np.abs(gp[:, 2]) < ZLIM)
gp, gm, feh = gp[ins], gm[ins], feh[ins]
fin = np.isfinite(feh)
rng = [[-XLIM, XLIM], [-ZLIM, ZLIM]]
W, xe, ze = np.histogram2d(gp[:, 0], gp[:, 2], bins=NB, range=rng, weights=gm)
den = np.histogram2d(gp[fin, 0], gp[fin, 2], bins=NB, range=rng, weights=gm[fin])[0]
num = np.histogram2d(gp[fin, 0], gp[fin, 2], bins=NB, range=rng,
                     weights=gm[fin] * feh[fin])[0]
M = np.where(den > MMIN, num / np.where(den > 0, den, 1), np.nan)
AREA = float((xe[1] - xe[0]) * (ze[1] - ze[0]))
SIG = np.where(W > 0, W / AREA, np.nan)
print(f'  pixels with SF gas: {100 * np.isfinite(SIG).mean():.2f} %; '
      f'with a drawn [Fe/H]: {100 * np.isfinite(M).mean():.2f} % '
      f'(holding {100 * den[np.isfinite(M)].sum() / den.sum():.1f} % of the SF gas)')

# ---------------------------------- figure -----------------------------------
FW = 7.8
AXL, AXW = .115, .715
axw_in = FW * AXW
axh_in = axw_in * ZLIM / XLIM
FH = 2 * axh_in + 1.02
fig = plt.figure(figsize=(FW, FH))
b0, h = .055 * (9.8 / FH), axh_in / FH
axes = [fig.add_axes([AXL, b0 + h, AXW, h]), fig.add_axes([AXL, b0, AXW, h])]
GAP = .012
cax_a = fig.add_axes([AXL + AXW + .018, b0 + h + GAP, .026, h - GAP])
cax_b = fig.add_axes([AXL + AXW + .018, b0, .026, h - GAP])

ax = axes[0]
im = ax.pcolormesh(xe, ze, SIG.T, cmap='Greys',
                   norm=LogNorm(vmin=SIG_LO, vmax=SIG_HI), rasterized=True)
cb = fig.colorbar(im, cax=cax_a)
cb.set_label(r'$\Sigma_{\rm gas,\,SF}$ [M$_\odot$ kpc$^{-2}$]')
OT.density_contours(ax, G[:, 0], G[:, 2], rng, '#8E24AA',
                    levels=(0.9, 0.5), bins=70, smooth=1.6, lw=2.2)
ax.plot([], [], color='#8E24AA', lw=2.2, label='GS/E stellar debris')
ax.plot([0, gc_[0]], [0, gc_[2]], color='k', ls='--', lw=1.8, alpha=.85,
        label='the lane')
ax.legend(loc='lower right', handlelength=1.5, borderpad=.35)
ax.text(.03, .965, '(a)', transform=ax.transAxes, va='top', fontsize=16,
        fontweight='bold')
ax.text(.5, .965, f'star-forming gas only\nsnapshot {SNAP},  $t$ = {TSNAP:.2f} Gyr',
        transform=ax.transAxes, va='top', ha='center', fontsize=12,
        bbox=dict(fc='white', ec='none', alpha=.8, pad=2))

ax = axes[1]
im = ax.pcolormesh(xe, ze, M.T, cmap='viridis', vmin=VMIN, vmax=VMAX, rasterized=True)
cb = fig.colorbar(im, cax=cax_b)
cb.set_label('[Fe/H] of the star-forming gas')
OT.density_contours(ax, G[:, 0], G[:, 2], rng, 'w',
                    levels=(0.9, 0.5), bins=70, smooth=1.6, lw=2.2)
ax.plot([0, gc_[0]], [0, gc_[2]], color='w', ls='--', lw=1.8, alpha=.9)
ax.text(.03, .965, '(b)', transform=ax.transAxes, va='top', fontsize=16,
        fontweight='bold')

for a_ in axes:
    a_.set(aspect='equal', xlim=(-XLIM, XLIM), ylim=(-ZLIM, ZLIM), ylabel='$z$ [kpc]')
axes[0].tick_params(labelbottom=False)
axes[1].set_xlabel('$x$ [kpc]')
keep = [t for t in axes[0].get_yticks() if -ZLIM + 1 < t < ZLIM - 1]
axes[0].set_yticks(keep); axes[0].set_ylim(-ZLIM, ZLIM)
f = f'{OUT}/au18_gas_metallicity_sf_snap{SNAP}.png'
fig.savefig(f, bbox_inches='tight')
print(f'\nsaved {f}')
