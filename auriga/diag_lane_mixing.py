"""Is the lane gas MIXED, or two unmixed components in varying proportion?

Two claims made in this project sound contradictory:

  A  (au18_gas_metallicity)  the lane is "the host's gas on a metallicity
     gradient, not a distinct satellite reservoir": [Fe/H] falls smoothly along
     it with no discontinuity where a second reservoir would begin.

  B  (au18_nitrogen_dispersion)  stars born in the lane have a BIMODAL [N/Fe] at
     fixed [Fe/H], with the two modes sitting on host-like and GS/E-like values,
     i.e. the gas there is NOT mixed.

They are only contradictory if a smooth gradient in the MEAN implies mixing at
the cell level.  It does not: a two-component medium whose mixing FRACTION f
varies with position has mean Z = f Z_host + (1-f) Z_sat, which runs smoothly
from one end member to the other while every individual cell stays at one value
or the other.  Claim A constrains the first moment; claim B is about the second.

This measures both moments of the same gas, in 3D so that nothing is blamed on
projection.  Cells are binned along the host-satellite axis; in each bin:

  * mass-weighted mean and median [Fe/H]        -> claim A
  * the 16-84 spread, and a 2-component Gaussian mixture fitted by EM with a BIC
    test against one component                  -> claim B

If the mean runs smoothly AND the per-bin distribution is bimodal with the
component means pinned near the two end members, both claims hold and the
resolution is that the lane is macroscopically a mixing sequence and
microscopically unmixed.

The projected map cannot settle this.  Each of its pixels is a mass-weighted
mean through the whole box along y, so it reports the first moment by
construction and smooths any discontinuity -- which is exactly why claim A,
read off that map, is weaker evidence for mixing than it looks.

Writes figures/au18_lane_mixing.png
"""
import gc, os, sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import auriga_public as ap
import config_au18 as C
import au18_frame as AF

SNAP = 72
R_LANE = 3.5                      # cylinder radius about the host-satellite axis
TLO, THI, NT = 0.20, 0.90, 7      # fractional distance along the axis, in NT bins
NFIT = 200
OUT = C.FIG_DIR

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 13, 'axes.labelsize': 14.5,
    'xtick.labelsize': 12.5, 'ytick.labelsize': 12.5, 'legend.fontsize': 11,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'figure.dpi': 140, 'savefig.dpi': 200,
})


def em2(x, w, iters=400, tol=1e-10):
    """Mass-weighted 2-component 1D Gaussian mixture."""
    x = np.asarray(x, float); w = np.asarray(w, float) / np.sum(w)
    mu = np.array([np.percentile(x, 20), np.percentile(x, 80)])
    sd = np.full(2, max(np.std(x) / 2., 1e-3)); pi = np.array([.5, .5])
    ll_old = -np.inf
    for _ in range(iters):
        p = pi[None, :] * np.exp(-.5 * ((x[:, None] - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        tot = np.maximum(p.sum(1), 1e-300)
        r = (p / tot[:, None]) * w[:, None]
        ll = float(np.sum(w * np.log(tot)))
        nk = np.maximum(r.sum(0), 1e-12)
        pi = nk / nk.sum()
        mu = (r * x[:, None]).sum(0) / nk
        sd = np.maximum(np.sqrt((r * (x[:, None] - mu) ** 2).sum(0) / nk), 1e-3)
        if abs(ll - ll_old) < tol * max(abs(ll), 1.):
            break
        ll_old = ll
    o = np.argsort(mu)
    return pi[o], mu[o], sd[o], ll


def ll1(x, w):
    w = w / w.sum()
    m = float(np.sum(w * x)); s = max(float(np.sqrt(np.sum(w * (x - m) ** 2))), 1e-3)
    return float(np.sum(w * (-.5 * ((x - m) / s) ** 2 - np.log(s * np.sqrt(2 * np.pi)))))


def wq(x, w, q):
    i = np.argsort(x); x, w = x[i], w[i]
    c = np.cumsum(w)
    return np.interp(np.atleast_1d(q) * c[-1], c, x)


# ------------------------------- load ----------------------------------------
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
AX = (ROT @ np.median(G, axis=0))                 # the satellite, in the figure frame

g = ap.snapshot.load_snapshot(SNAP, 0, snappath=C.SIM_DIR, verbose=False,
    loadlist=['Coordinates', 'Masses', 'StarFormationRate', 'GFM_Metals'])
g = ap.util.CentreOnHalo(g, cen); ap.util.rotateto(g, xd, dir2=yd, dir3=zd)
cg = g.data['Coordinates']
gp = (np.column_stack([cg[:, 1], cg[:, 2], cg[:, 0]]) * 1e3) @ ROT.T
gm = g.data['Masses'] * C.MASS_TO_MSUN
SF = np.asarray(g.data['StarFormationRate'], float) > 0
met = g.data['GFM_Metals']
feh = C.bracket_abundance(met, 'Fe', 'H')
nfe = C.bracket_abundance(met, 'N', 'Fe')
del g, cg, met; gc.collect()

# 3D lane: a cylinder on the host-satellite axis.  No projection anywhere.
u = AX / np.linalg.norm(AX)
tpar = gp @ u / np.linalg.norm(AX)
perp = np.linalg.norm(gp - np.outer(tpar * np.linalg.norm(AX), u), axis=1)
lane = (perp < R_LANE) & (tpar > TLO) & (tpar < THI) & np.isfinite(feh) & np.isfinite(nfe)
print(f'snapshot {SNAP}: satellite at r = {np.linalg.norm(AX):.1f} kpc; '
      f'lane cylinder radius {R_LANE} kpc, t = {TLO}-{THI}')
print(f'lane cells {lane.sum():,} ({(lane & SF).sum():,} star-forming), '
      f'M = {gm[lane].sum():.3e} Msun ({gm[lane & SF].sum():.3e} SF)\n')

edges = np.linspace(TLO, THI, NT + 1)
print(f'{"t":>12s} {"R[kpc]":>7s} {"N":>7s} {"M_gas":>10s} {"mean":>7s} {"med":>7s} '
      f'{"16-84":>7s} | {"w_lo":>5s} {"mu_lo":>7s} {"mu_hi":>7s} {"sep":>6s} {"dBIC":>8s}')
rows = []
for i in range(NT):
    m = lane & (tpar >= edges[i]) & (tpar < edges[i + 1])
    if m.sum() < NFIT:
        continue
    x, w = feh[m], gm[m]
    mean = float(np.sum(w * x) / w.sum())
    q16, q50, q84 = wq(x, w, [.16, .5, .84])
    pi, mu, sd, l2 = em2(x, w)
    db = (2 * np.log(m.sum()) - 2 * ll1(x, w) * m.sum()) - \
         (5 * np.log(m.sum()) - 2 * l2 * m.sum())
    R = .5 * (edges[i] + edges[i + 1]) * np.linalg.norm(AX)
    rows.append((.5 * (edges[i] + edges[i + 1]), R, m.sum(), w.sum(), mean, q50,
                 q84 - q16, pi[0], mu[0], mu[1], mu[1] - mu[0], db))
    print(f'{edges[i]:.2f}-{edges[i+1]:.2f} {R:>7.1f} {m.sum():>7,} {w.sum():>10.3e} '
          f'{mean:>+7.3f} {q50:>+7.3f} {q84 - q16:>7.3f} | {pi[0]:>5.2f} {mu[0]:>+7.3f} '
          f'{mu[1]:>+7.3f} {mu[1] - mu[0]:>6.3f} {db:>8.0f}')
R = np.array(rows, float)

# ---------------------------------- figure -----------------------------------
# The decisive comparison with the stars: THEY were bimodal in [N/Fe] AT FIXED
# [Fe/H], which is a different cut from the unconditioned [Fe/H] distribution
# panel (c) shows.  Ask the gas the same question.
FB = (-0.45, -0.25)
mb = lane & (feh >= FB[0]) & (feh < FB[1])
pi_n, mu_n, sd_n, l2n = em2(nfe[mb], gm[mb])
dbn = (2 * np.log(mb.sum()) - 2 * ll1(nfe[mb], gm[mb]) * mb.sum()) - \
      (5 * np.log(mb.sum()) - 2 * l2n * mb.sum())
print(f'\nlane gas at {FB[0]} < [Fe/H] < {FB[1]}: {mb.sum():,} cells, '
      f'M = {gm[mb].sum():.3e} Msun')
print(f'  [N/Fe] 2-component EM: w = {pi_n[0]:.2f}/{pi_n[1]:.2f}, '
      f'mu = {mu_n[0]:+.3f}/{mu_n[1]:+.3f}, separation {mu_n[1]-mu_n[0]:.3f} dex, '
      f'sd = {sd_n[0]:.3f}/{sd_n[1]:.3f}, dBIC = {dbn:.0f}')
q = wq(nfe[mb], gm[mb], [.16, .5, .84])
print(f'  [N/Fe] 16/50/84 = {q[0]:+.3f} / {q[1]:+.3f} / {q[2]:+.3f}  '
      f'(16-84 width {q[2]-q[0]:.3f} dex)')
# the same for the two ends of the lane, to see whether the mix changes
for nm, tm in (('inner half', tpar < .55), ('outer half', tpar >= .55)):
    k = mb & tm
    if k.sum() < NFIT:
        continue
    qq = wq(nfe[k], gm[k], [.16, .5, .84])
    print(f'  {nm}: N = {k.sum():>6,}, median [N/Fe] = {qq[1]:+.3f}, '
          f'16-84 = {qq[2]-qq[0]:.3f} dex')

fig, AXS = plt.subplots(1, 4, figsize=(21.0, 5.0))
ax = AXS[0]
ax.plot(R[:, 1], R[:, 4], 'o-', color='k', lw=2.4, ms=6, label='mass-weighted mean')
ax.fill_between(R[:, 1], R[:, 4] - .5 * R[:, 6], R[:, 4] + .5 * R[:, 6],
                color='k', alpha=.15, lw=0, label='16$-$84 of the cells')
ax.plot(R[:, 1], R[:, 8], 's--', color='#4C72B0', lw=2.0, ms=5,
        label='metal-poor component')
ax.plot(R[:, 1], R[:, 9], 's--', color='#C44E52', lw=2.0, ms=5,
        label='metal-rich component')
ax.set(xlabel='distance along the lane [kpc]', ylabel='[Fe/H] of the gas')
ax.legend(loc='upper right', fontsize=10.5)
ax.text(.03, .96, '(a)  the mean runs smoothly', transform=ax.transAxes, va='top',
        fontsize=12.5, fontweight='bold')

ax = AXS[1]
ax.plot(R[:, 1], R[:, 10], 'o-', color='k', lw=2.4, ms=6, label='component separation')
ax.plot(R[:, 1], R[:, 6], 's--', color='.45', lw=2.0, ms=5, label='16$-$84 spread')
ax.set(xlabel='distance along the lane [kpc]', ylabel='[Fe/H] width [dex]',
       ylim=(0, None))
ax.legend(loc='upper right', fontsize=10.5)
ax.text(.03, .96, '(b)  but the cells are not one population',
        transform=ax.transAxes, va='top', fontsize=12.5, fontweight='bold')

ax = AXS[2]
bins = np.linspace(-1.2, .3, 90)
cmap = plt.get_cmap('viridis')
for k, i in enumerate(range(len(rows))):
    m = lane & (tpar >= edges[i]) & (tpar < edges[i + 1])
    ax.hist(feh[m], bins=bins, weights=gm[m], density=True, histtype='step',
            color=cmap(k / max(len(rows) - 1, 1)), lw=2.0,
            label=f'{R[i, 1]:.0f} kpc')
ax.set(xlabel='[Fe/H] of the gas', ylabel='mass-weighted density')
ax.legend(loc='upper left', fontsize=10, title='along the lane', title_fontsize=10)
ax.text(.03, .96, '(c)  the distribution in each bin', transform=ax.transAxes,
        va='top', fontsize=12.5, fontweight='bold', ha='left')

ax = AXS[3]
nb = np.linspace(-.15, .30, 80)
ax.hist(nfe[mb], bins=nb, weights=gm[mb], density=True, histtype='stepfilled',
        color='#00897B', alpha=.30, lw=0)
ax.hist(nfe[mb], bins=nb, weights=gm[mb], density=True, histtype='step',
        color='#00897B', lw=2.4, label=f'lane gas ({mb.sum():,} cells)')
xx = np.linspace(-.15, .30, 400)
tot = np.zeros_like(xx)
for j, col in enumerate(('#4C72B0', '#C44E52')):
    cur = pi_n[j] * np.exp(-.5 * ((xx - mu_n[j]) / sd_n[j]) ** 2) / (sd_n[j] * np.sqrt(2 * np.pi))
    tot += cur
    ax.plot(xx, cur, color=col, lw=2.2,
            label=f'$\\mu$ = {mu_n[j]:+.3f}, $w$ = {pi_n[j]:.2f}')
ax.plot(xx, tot, 'k--', lw=1.6, label='sum')
ax.set(xlabel='[N/Fe] of the gas', ylabel='mass-weighted density',
       title=f'${FB[0]} <$ [Fe/H] $< {FB[1]}$')
ax.legend(loc='upper left', fontsize=10)
ax.text(.03, .74, '(d)  the same cut the stars showed', transform=ax.transAxes,
        va='top', fontsize=12.5, fontweight='bold')

fig.tight_layout(pad=.6, w_pad=1.2)
f = f'{OUT}/au18_lane_mixing.png'
fig.savefig(f, bbox_inches='tight')
print(f'\nsaved {f}')
