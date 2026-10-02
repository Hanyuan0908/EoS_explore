"""Is there evidence that the lane gas is MIXING, or only that two supplies sit
side by side in varying proportion?

Everything shown so far -- a smooth mean gradient, a large internal spread, a
bimodal [N/Fe] at fixed [Fe/H] -- is equally well explained by JUXTAPOSITION:
pure host parcels and pure satellite parcels interleaved, with the proportion
changing along the lane and no material ever exchanged between them.  Mixing
means something stronger: cells acquiring compositions INTERMEDIATE between the
two end members, compositions that neither supply contains.

THE TEST.  Measure the two end members' own [Fe/H] distributions, including
their internal spread, then ask whether each lane bin can be written as a
non-negative mixture of them,

    P_bin(Z)  ~=  f P_host(Z)  +  (1 - f) P_sat(Z)

with a single free f.  Under juxtaposition this must fit: every parcel in the
lane came from one end or the other, so the lane distribution is literally a
weighted sum of the two.  Mixing shows up as mass the fit cannot place --
cells at metallicities that are rare in BOTH end members.

End members are taken from the same cylinder so the geometry cannot bias them:

    host   the ANTI-LANE: the mirror of the same cylinder through the centre,
           -0.9 < t < -0.2, so it samples host gas at the SAME galactocentric
           radii as the lane but on the opposite side, where no satellite
           material has reached
    sat    within 3 kpc of the GS/E centroid

An earlier version took the host end member from the foot of the lane itself
(0.02 < t < 0.15).  That was wrong: at |r_GSE| = 19.7 kpc those t values are
0.4-3.0 kpc from the centre, i.e. the nucleus, whose median [Fe/H] = +0.09 is
nothing like the host gas that actually feeds a lane at 10 kpc.  Using it made
every mixture predict a metal-rich hump the lane does not have, and the fit
residual was 0.08-0.14 throughout.  The anti-lane is matched in radius and is
the honest comparison.

The second statistic is assumption-free: the fraction of lane mass lying strictly
between the 84th percentile of the satellite end member and the 16th percentile
of the host end member -- the gap that juxtaposition leaves empty.

CAVEAT ON THE CODE, which bounds how much mixing is even possible.  Arepo
advects metals with the gas and has no explicit metal-diffusion term, so
composition is exchanged only through resolved mass fluxes between cells and
through refinement/derefinement.  Whatever mixing this measures is therefore a
LOWER bound on the physical rate, and a reader should not read the number as a
measurement of real ISM mixing.

Writes figures/au18_lane_mixing_test.png
"""
import gc, os, sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import auriga_public as ap
import config_au18 as C
import au18_frame as AF

SNAP = int(sys.argv[1]) if len(sys.argv) > 1 else 72
R_LANE, TLO, THI, NT = 3.5, 0.20, 0.90, 7
TH_LO, TH_HI = -0.90, -0.20        # host end member: the anti-lane, matched in radius
R_SAT = 3.0                        # satellite end member
BINS = np.linspace(-1.3, 0.4, 120)
OUT = C.FIG_DIR

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 13, 'axes.labelsize': 14.5,
    'xtick.labelsize': 12.5, 'ytick.labelsize': 12.5, 'legend.fontsize': 10.5,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'figure.dpi': 140, 'savefig.dpi': 200,
})


def wq(x, w, q):
    i = np.argsort(x); x, w = x[i], w[i]
    c = np.cumsum(w)
    return np.interp(np.atleast_1d(q) * c[-1], c, x)


def norm_hist(x, w):
    h = np.histogram(x, bins=BINS, weights=w)[0].astype(float)
    return h / max(h.sum(), 1e-30)


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
AXv = ROT @ np.median(G, axis=0)
Lax = float(np.linalg.norm(AXv)); u = AXv / Lax

g = ap.snapshot.load_snapshot(SNAP, 0, snappath=C.SIM_DIR, verbose=False,
    loadlist=['Coordinates', 'Masses', 'GFM_Metals'])
g = ap.util.CentreOnHalo(g, cen); ap.util.rotateto(g, xd, dir2=yd, dir3=zd)
cg = g.data['Coordinates']
gp = (np.column_stack([cg[:, 1], cg[:, 2], cg[:, 0]]) * 1e3) @ ROT.T
gm = g.data['Masses'] * C.MASS_TO_MSUN
MET = np.asarray(g.data['GFM_Metals'])
feh = C.bracket_abundance(MET, 'Fe', 'H')
del g, cg; gc.collect()

tpar = (gp @ u) / Lax
perp = np.linalg.norm(gp - np.outer(tpar * Lax, u), axis=1)
fin = np.isfinite(feh)
cyl = (perp < R_LANE) & fin
EH = cyl & (tpar > TH_LO) & (tpar < TH_HI)      # the anti-lane
ES = fin & (np.linalg.norm(gp - AXv, axis=1) < R_SAT)
PH, PS = norm_hist(feh[EH], gm[EH]), norm_hist(feh[ES], gm[ES])
qh = wq(feh[EH], gm[EH], [.16, .5, .84])
qs = wq(feh[ES], gm[ES], [.16, .5, .84])
print(f'end members at snapshot {SNAP}:')
print(f'  host  (anti-lane, t = {TH_LO} to {TH_HI}): {EH.sum():>6,} cells, '
      f'M = {gm[EH].sum():.3e}, [Fe/H] 16/50/84 = {qh[0]:+.3f}/{qh[1]:+.3f}/{qh[2]:+.3f}')
print(f'  sat   (< {R_SAT:.0f} kpc of the centroid): {ES.sum():>6,} cells, '
      f'M = {gm[ES].sum():.3e}, [Fe/H] 16/50/84 = {qs[0]:+.3f}/{qs[1]:+.3f}/{qs[2]:+.3f}')
GAP = (qs[2], qh[0])
print(f'  the gap juxtaposition cannot fill: {GAP[0]:+.3f} < [Fe/H] < {GAP[1]:+.3f}')
if GAP[0] >= GAP[1]:
    print('  !! the end members overlap; the gap test is vacuous at this snapshot')

ctrB = .5 * (BINS[:-1] + BINS[1:])
inG = (ctrB > GAP[0]) & (ctrB < GAP[1])
fg_h, fg_s = PH[inG].sum(), PS[inG].sum()
print(f'  mass fraction of each END MEMBER already in that gap: '
      f'host {100 * fg_h:.1f} %, satellite {100 * fg_s:.1f} %')

edges = np.linspace(TLO, THI, NT + 1)
print(f'\n{"R[kpc]":>7s} {"N":>7s} {"f_host":>7s} {"resid":>7s} '
      f'{"in gap":>8s} {"expected":>9s} {"excess":>8s}')
rows = []
for i in range(NT):
    m = cyl & (tpar >= edges[i]) & (tpar < edges[i + 1])
    if m.sum() < 200:
        continue
    P = norm_hist(feh[m], gm[m])
    ff = np.linspace(0, 1, 501)
    res = [np.sum((P - (a * PH + (1 - a) * PS)) ** 2) for a in ff]
    f_best = float(ff[int(np.argmin(res))])
    model = f_best * PH + (1 - f_best) * PS
    obs_gap = float(P[inG].sum())
    exp_gap = float(model[inG].sum())
    R = .5 * (edges[i] + edges[i + 1]) * Lax
    rows.append((R, m.sum(), f_best, float(np.sqrt(np.min(res))), obs_gap, exp_gap,
                 obs_gap - exp_gap, P, model))
    print(f'{R:>7.1f} {m.sum():>7,} {f_best:>7.2f} {np.sqrt(np.min(res)):>7.4f} '
          f'{100 * obs_gap:>7.1f}% {100 * exp_gap:>8.1f}% {100 * (obs_gap - exp_gap):>7.1f}%')

# ------------- the same test in [N/Fe], at fixed [Fe/H] ----------------------
# [Fe/H] turned out to be useless for this: the host's own gas at matched radii
# spans 0.94 dex (16-84) and swallows the satellite's whole range, so there is no
# gap to look in.  The reservoirs DO separate in [N/Fe] at fixed [Fe/H] -- that is
# the cut the stellar result was made in -- so repeat the test there.
FSL = (-0.60, -0.40)
NBINS = np.linspace(-0.10, 0.30, 90)
nfe = C.bracket_abundance(MET, 'N', 'Fe')
nfin = np.isfinite(nfe) & np.isfinite(feh) & (feh >= FSL[0]) & (feh < FSL[1])


def nhist(m):
    h = np.histogram(nfe[m], bins=NBINS, weights=gm[m])[0].astype(float)
    return h / max(h.sum(), 1e-30)


EHn, ESn = EH & nfin, ES & nfin
print(f'\n[N/Fe] test inside {FSL[0]} < [Fe/H] < {FSL[1]}:')
if EHn.sum() < 100 or ESn.sum() < 100:
    print(f'  too few cells in the slice (host {EHn.sum()}, sat {ESn.sum()}); skipped')
else:
    PHn, PSn = nhist(EHn), nhist(ESn)
    qhn = wq(nfe[EHn], gm[EHn], [.16, .5, .84])
    qsn = wq(nfe[ESn], gm[ESn], [.16, .5, .84])
    print(f'  host  {EHn.sum():>6,} cells, [N/Fe] 16/50/84 = '
          f'{qhn[0]:+.3f}/{qhn[1]:+.3f}/{qhn[2]:+.3f}')
    print(f'  sat   {ESn.sum():>6,} cells, [N/Fe] 16/50/84 = '
          f'{qsn[0]:+.3f}/{qsn[1]:+.3f}/{qsn[2]:+.3f}')
    GN = (min(qhn[2], qsn[2]), max(qhn[0], qsn[0]))
    ctrN = .5 * (NBINS[:-1] + NBINS[1:])
    inGN = (ctrN > GN[0]) & (ctrN < GN[1])
    if GN[0] >= GN[1] or inGN.sum() == 0:
        print(f'  end members overlap in [N/Fe] too ({GN[0]:+.3f} to {GN[1]:+.3f}); '
              f'no gap, test vacuous')
    else:
        print(f'  gap {GN[0]:+.3f} < [N/Fe] < {GN[1]:+.3f}; end members already there: '
              f'host {100*PHn[inGN].sum():.1f} %, sat {100*PSn[inGN].sum():.1f} %')
        print(f'{"R[kpc]":>7s} {"N":>7s} {"f_host":>7s} {"in gap":>8s} '
              f'{"expected":>9s} {"excess":>8s}')
        for i in range(NT):
            m = cyl & nfin & (tpar >= edges[i]) & (tpar < edges[i + 1])
            if m.sum() < 200:
                continue
            P = nhist(m)
            ff = np.linspace(0, 1, 501)
            res = [np.sum((P - (a * PHn + (1 - a) * PSn)) ** 2) for a in ff]
            fb = float(ff[int(np.argmin(res))])
            mod = fb * PHn + (1 - fb) * PSn
            R_ = .5 * (edges[i] + edges[i + 1]) * Lax
            print(f'{R_:>7.1f} {m.sum():>7,} {fb:>7.2f} {100*P[inGN].sum():>7.1f}% '
                  f'{100*mod[inGN].sum():>8.1f}% {100*(P[inGN].sum()-mod[inGN].sum()):>7.1f}%')

# ---------------------------------- figure -----------------------------------
fig, AXS = plt.subplots(1, 3, figsize=(16.5, 5.0))
ax = AXS[0]
ax.step(ctrB, PH, where='mid', color='#C44E52', lw=2.4,
        label='host end member (anti-lane)')
ax.step(ctrB, PS, where='mid', color='#4C72B0', lw=2.4, label='satellite end member')
ax.axvspan(GAP[0], GAP[1], color='0.75', alpha=.45, lw=0, label='the gap')
ax.set(xlabel='[Fe/H] of the gas', ylabel='mass fraction per bin', xlim=(-1.3, .4))
ax.legend(loc='upper left')
ax.text(.03, .74, '(a)  what the two supplies look like', transform=ax.transAxes,
        va='top', fontsize=12.5, fontweight='bold')

ax = AXS[1]
cmap = plt.get_cmap('viridis')
for k, r in enumerate(rows):
    col = cmap(k / max(len(rows) - 1, 1))
    ax.step(ctrB, r[7], where='mid', color=col, lw=2.0, label=f'{r[0]:.0f} kpc')
    ax.step(ctrB, r[8], where='mid', color=col, lw=1.0, ls='--', alpha=.8)
ax.axvspan(GAP[0], GAP[1], color='0.75', alpha=.45, lw=0)
ax.set(xlabel='[Fe/H] of the gas', ylabel='mass fraction per bin', xlim=(-1.3, .4))
ax.legend(loc='upper left', ncol=2, fontsize=9.5, title='solid: lane\ndashed: best mixture',
          title_fontsize=9.5)
ax.text(.03, .74, '(b)  can a mixture of them fit?', transform=ax.transAxes,
        va='top', fontsize=12.5, fontweight='bold')

ax = AXS[2]
RR = np.array([r[0] for r in rows])
ax.plot(RR, 100 * np.array([r[4] for r in rows]), 'o-', color='k', lw=2.4, ms=6,
        label='observed in the gap')
ax.plot(RR, 100 * np.array([r[5] for r in rows]), 's--', color='.5', lw=2.0, ms=5,
        label='juxtaposition predicts')
ax.fill_between(RR, 100 * np.array([r[5] for r in rows]),
                100 * np.array([r[4] for r in rows]), color='#00897B', alpha=.25,
                lw=0, label='excess = mixed material')
ax.set(xlabel='distance along the lane [kpc]',
       ylabel='per cent of the bin mass in the gap', ylim=(0, None))
ax.legend(loc='upper right')
ax.text(.03, .96, '(c)  the material neither supply has', transform=ax.transAxes,
        va='top', fontsize=12.5, fontweight='bold')

fig.tight_layout(pad=.6, w_pad=1.2)
f = f'{OUT}/au18_lane_mixing_test_snap{SNAP}.png'
fig.savefig(f, bbox_inches='tight')
print(f'\nsaved {f}')
