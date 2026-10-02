"""What the two [N/Fe] components in the bridge actually are.

diag_nfe_bridge_trend showed the bridge [N/Fe] distribution splits into two
Gaussians whose separation Delta drives the sigma_[N/Fe] trend.  It did not show
what the two components ARE.  This does, by taking the peak bin
([Fe/H] = -0.4 to -0.2, where the split is cleanest: dBIC = 351, Delta = 0.100),
assigning stars to a component by EM posterior > PCUT, and then asking four
questions of the two groups:

 1  is the split N-only?      compare median [X/Fe] for C, N, O, Ne, Mg, Si.
                              N and C come largely from AGB stars, delayed by
                              ~10^8-10^9 yr; O, Ne, Mg, Si come promptly from
                              core collapse.  A split confined to N and C means
                              the two gas supplies differ in how much DELAYED
                              enrichment they have had, not in how much metal.
                              A split in every element means two reservoirs of
                              wholly different composition.

 2  are they in different      distance from the GS/E centroid and from the host
    places?                    centre, within the bridge.  Two gas streams that
                               have not mixed should be spatially separated;
                               one turbulent medium should not be.

 3  which one is GS/E-like?    compare both against the clean GS/E debris and
                               against host-aperture stars at the same [Fe/H].

 4  is it secondary nitrogen?  [N/O] against [O/H].  Secondary N rises with
                               metallicity; primary N is flat.

Writes figures/au18_nfe_bimodality.png
"""
import gc, os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import auriga_public as ap
import config_au18 as C
import au18_frame as AF

SNAP = 72
CUT, ZCUT = 0.8, 1.5
FLO, FHI = -0.4, -0.2          # the peak bin, where the split is cleanest
PCUT = 0.8                     # posterior purity for assigning a component
R_SAT, R_HOST = 6.0, 8.0
ELS = ['C', 'N', 'O', 'Ne', 'Mg', 'Si']
NBOOT = 500
RNG = np.random.default_rng(0)
C1, C2 = '#4C72B0', '#C44E52'
cSAT, cHOST, cGSE = '#8E24AA', '#B8860B', '#2B2B2B'
OUT = os.path.dirname(os.path.abspath(__file__)) + '/figures'

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 13, 'axes.labelsize': 14.5,
    'xtick.labelsize': 12.5, 'ytick.labelsize': 12.5, 'legend.fontsize': 11,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 140, 'savefig.dpi': 200,
})


def mad_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def med_err(x, n=NBOOT):
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan, np.nan
    b = np.median(RNG.choice(x, (n, len(x))), axis=1)
    return float(np.median(x)), float(np.std(b))


def em2(x, iters=500, tol=1e-10):
    x = np.asarray(x, float)
    mu = np.array([np.percentile(x, 20), np.percentile(x, 80)])
    s = np.full(2, max(np.std(x) / 2., 1e-3)); w = np.array([.5, .5])
    ll_old = -np.inf
    for _ in range(iters):
        p = w[None, :] * np.exp(-.5 * ((x[:, None] - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))
        tot = np.maximum(p.sum(1), 1e-300)
        r = p / tot[:, None]
        ll = float(np.log(tot).sum())
        nk = np.maximum(r.sum(0), 1e-10)
        w = nk / len(x); mu = (r * x[:, None]).sum(0) / nk
        s = np.maximum(np.sqrt((r * (x[:, None] - mu) ** 2).sum(0) / nk), 1e-3)
        if abs(ll - ll_old) < tol * max(abs(ll), 1.):
            break
        ll_old = ll
    o = np.argsort(mu)
    return w[o], mu[o], s[o], r[:, o]


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
pg = np.searchsorted(ss, GSE)
okg = (pg < len(ss)) & (ss[np.minimum(pg, len(ss) - 1)] == GSE)
Mg_ = met[o[pg[okg]]]
del met, pos; gc.collect()

feh = C.bracket_abundance(M, 'Fe', 'H')
XFE = {e: C.bracket_abundance(M, e, 'Fe') for e in ELS}
oh = C.bracket_abundance(M, 'O', 'H')
no = C.bracket_abundance(M, 'N', 'O')
feh_g = C.bracket_abundance(Mg_, 'Fe', 'H')
XFE_g = {e: C.bracket_abundance(Mg_, e, 'Fe') for e in ELS}
oh_g = C.bracket_abundance(Mg_, 'O', 'H')
no_g = C.bracket_abundance(Mg_, 'N', 'O')

halo = ~disc
d_gse = np.linalg.norm(P - g2, axis=1)
d_host = np.linalg.norm(P, axis=1)
sat = d_gse < R_SAT
host = (~sat) & (d_host < R_HOST)
bridge = ~(sat | host)
inbin = (feh >= FLO) & (feh < FHI)
BR = bridge & halo & np.isfinite(XFE['N']) & inbin

# ------------------------- split the bridge by EM ----------------------------
y = XFE['N'][BR]
w, mu, sd, post = em2(y)
idxBR = np.flatnonzero(BR)
A = np.zeros(len(feh), bool); B = np.zeros(len(feh), bool)
A[idxBR[post[:, 0] > PCUT]] = True      # N-poor component
B[idxBR[post[:, 1] > PCUT]] = True      # N-rich component
print(f'peak bin [Fe/H] = {FLO} to {FHI}, bridge halo-born N = {BR.sum():,}')
print(f'  EM: w = {w[0]:.2f}/{w[1]:.2f}, mu = {mu[0]:+.3f}/{mu[1]:+.3f}, '
      f'Delta = {mu[1]-mu[0]:.3f}, s = {sd[0]:.3f}/{sd[1]:.3f}')
print(f'  assigned at posterior > {PCUT}: N-poor {A.sum():,}, N-rich {B.sum():,}, '
      f'ambiguous {BR.sum()-A.sum()-B.sum():,}')

SAT = sat & halo & inbin
HOST = host & halo & inbin
GDEB = np.isfinite(feh_g) & (feh_g >= FLO) & (feh_g < FHI)
print(f'  reference groups in the same bin: satellite {SAT.sum():,}, '
      f'host {HOST.sum():,}, GS/E debris {GDEB.sum():,}')

# --- question 1: is the split N-only? ---
print('\nQ1  median [X/Fe] of the two components (bootstrap error on the median)')
print(f'{"el":>3s} {"N-poor":>16s} {"N-rich":>16s} {"difference":>16s} {"n sigma":>8s}')
diff, derr = {}, {}
for e in ELS:
    m1, e1 = med_err(XFE[e][A]); m2, e2 = med_err(XFE[e][B])
    d, de = m2 - m1, np.hypot(e1, e2)
    diff[e], derr[e] = d, de
    print(f'{e:>3s} {m1:>+10.4f}+-{e1:<5.4f} {m2:>+10.4f}+-{e2:<5.4f} '
          f'{d:>+10.4f}+-{de:<5.4f} {abs(d)/de:>8.1f}')

# control: the two components must be at the SAME [Fe/H], or the abundance
# differences above are just a residual metallicity offset inside the bin
mA, eA = med_err(feh[A]); mB, eB = med_err(feh[B])
print(f'    control, median [Fe/H] in the bin: N-poor {mA:+.4f}+-{eA:.4f}, '
      f'N-rich {mB:+.4f}+-{eB:.4f}, offset {mB-mA:+.4f}')

# --- question 2: are they in different places? ---
print('\nQ2  position within the bridge')
for nm, m in (('N-poor', A), ('N-rich', B)):
    print(f'  {nm:7s} d(GS/E) = {np.median(d_gse[m]):>5.1f} kpc '
          f'[{np.percentile(d_gse[m],16):.1f}-{np.percentile(d_gse[m],84):.1f}], '
          f'd(host) = {np.median(d_host[m]):>5.1f} kpc '
          f'[{np.percentile(d_host[m],16):.1f}-{np.percentile(d_host[m],84):.1f}]')

# --- question 3: which is GS/E-like? ---
print('\nQ3  [N/Fe] at the same [Fe/H]: where do the components sit?')
for nm, v in (('bridge N-poor', XFE['N'][A]), ('bridge N-rich', XFE['N'][B]),
              ('satellite site', XFE['N'][SAT]), ('host site', XFE['N'][HOST]),
              ('GS/E debris', XFE_g['N'][GDEB])):
    m_, e_ = med_err(v)
    print(f'  {nm:16s} {m_:>+8.4f} +- {e_:.4f}   (n = {np.isfinite(v).sum():,})')

# --- question 4: secondary nitrogen? ---
print('\nQ4  [N/O] vs [O/H], all bridge halo-born stars (not just this bin)')
BRall = bridge & halo & np.isfinite(no) & np.isfinite(oh)
qs = np.percentile(oh[BRall], [10, 30, 50, 70, 90])
for lo_, hi_ in zip(qs[:-1], qs[1:]):
    b = BRall & (oh >= lo_) & (oh < hi_)
    print(f'  [O/H] {lo_:+.2f} to {hi_:+.2f}:  median [N/O] = {np.median(no[b]):+.3f} '
          f'(n = {b.sum():,})')
sl = np.polyfit(oh[BRall], no[BRall], 1)[0]
print(f'  slope d[N/O]/d[O/H] = {sl:+.3f}   (0 = primary N, >0 = secondary N)')

# ---------------------------------- figure -----------------------------------
fig, AX = plt.subplots(2, 3, figsize=(16.5, 9.4))


def tag(ax, t, y=.96, ha='left', x=.03):
    ax.text(x, y, t, transform=ax.transAxes, va='top', ha=ha, fontsize=12.5,
            fontweight='bold')


ax = AX[0, 0]
bn = np.arange(-.10, .281, .012)
ax.hist(XFE['N'][BR], bins=bn, density=True, histtype='step', color='0.35', lw=2.0,
        label=f'all bridge ({BR.sum():,})')
ax.hist(XFE['N'][A], bins=bn, density=True, weights=np.full(A.sum(), w[0]),
        histtype='stepfilled', color=C1, alpha=.45, lw=0,
        label=f'N-poor, $p>{PCUT}$ ({A.sum():,})')
ax.hist(XFE['N'][B], bins=bn, density=True, weights=np.full(B.sum(), w[1]),
        histtype='stepfilled', color=C2, alpha=.45, lw=0,
        label=f'N-rich, $p>{PCUT}$ ({B.sum():,})')
ax.set(xlabel='[N/Fe]', ylabel='density', xlim=(-.10, .28))
ax.legend(loc='upper left', fontsize=10)
tag(ax, '(a)  the split being tested', y=.74)

ax = AX[0, 1]
xp = np.arange(len(ELS))
ax.errorbar(xp, [diff[e] for e in ELS], yerr=[derr[e] for e in ELS], fmt='o',
            color='k', ms=8, capsize=4, lw=1.6)
ax.axhline(0, color='0.5', lw=1.2)
ax.axhspan(-.01, .01, color='0.85', alpha=.6, lw=0)
for i, e in enumerate(ELS):
    ax.annotate(f'{abs(diff[e])/derr[e]:.0f}$\\sigma$', (xp[i], diff[e]),
                textcoords='offset points', xytext=(0, 13), ha='center', fontsize=10.5)
ax.set_xticks(xp); ax.set_xticklabels(ELS)
ax.set(xlabel='element X', ylabel=r'median [X/Fe]: N-rich $-$ N-poor [dex]',
       xlim=(-.6, len(ELS) - .4))
tag(ax, '(b)  Q1: is the split N-only?')

ax = AX[0, 2]
for nm, v, col in (('bridge N-poor', XFE['N'][A], C1),
                   ('bridge N-rich', XFE['N'][B], C2),
                   ('satellite site', XFE['N'][SAT], cSAT),
                   ('host site', XFE['N'][HOST], cHOST),
                   ('GS/E debris', XFE_g['N'][GDEB], cGSE)):
    m_, e_ = med_err(v)
    ax.errorbar(m_, nm, xerr=e_, fmt='o', color=col, ms=9, capsize=4, lw=1.8)
ax.axvline(mu[0], color=C1, lw=1.0, ls=':')
ax.axvline(mu[1], color=C2, lw=1.0, ls=':')
ax.set(xlabel=f'median [N/Fe] at [Fe/H] = {FLO} to {FHI}')
ax.grid(axis='x', alpha=.25)
tag(ax, '(c)  Q3: which component is GS/E-like?', y=.14)

ax = AX[1, 0]
ax.scatter(P[bridge & halo & inbin, 0], P[bridge & halo & inbin, 2], s=4,
           c='0.80', lw=0, rasterized=True)
ax.scatter(P[A, 0], P[A, 2], s=7, c=C1, alpha=.55, lw=0, rasterized=True,
           label='N-poor')
ax.scatter(P[B, 0], P[B, 2], s=7, c=C2, alpha=.55, lw=0, rasterized=True,
           label='N-rich')
th = np.linspace(0, 2 * np.pi, 200)
ax.plot(R_SAT * np.cos(th) + g2[0], R_SAT * np.sin(th) + g2[2], color=cSAT, lw=1.6)
ax.plot(R_HOST * np.cos(th), R_HOST * np.sin(th), color=cHOST, lw=1.6)
ax.set(aspect='equal', xlim=(-25, 25), ylim=(-25, 25), xlabel='$x$ [kpc]',
       ylabel='$z$ [kpc]')
ax.legend(loc='upper right', markerscale=2.2, fontsize=10.5)
tag(ax, '(d)  Q2: where are they?')

ax = AX[1, 1]
bd = np.arange(0, 41, 2)
for nm, m, col in (('N-poor', A, C1), ('N-rich', B, C2)):
    ax.hist(d_gse[m], bins=bd, density=True, histtype='step', color=col, lw=2.4,
            label=f'{nm}, median {np.median(d_gse[m]):.1f} kpc')
    ax.axvline(np.median(d_gse[m]), color=col, lw=1.1, ls='--')
ax.axvline(R_SAT, color=cSAT, lw=1.6)
ax.text(R_SAT + .6, .95, 'satellite aperture', transform=ax.get_xaxis_transform(),
        va='top', fontsize=10, color=cSAT, rotation=90)
ax.set(xlabel='distance from the GS/E centroid [kpc]', ylabel='density', xlim=(0, 40))
ax.legend(loc='upper right', fontsize=10.5)
tag(ax, '(e)  Q2: distance from GS/E')

ax = AX[1, 2]
BRall = bridge & halo & np.isfinite(no) & np.isfinite(oh)
h, xe, ye = np.histogram2d(oh[BRall], no[BRall], bins=(90, 90),
                           range=[[-1.4, .4], [-.2, .4]])
ax.pcolormesh(xe, ye, np.where(h > 0, h, np.nan).T, cmap='Greys', rasterized=True)
xx = np.linspace(-1.4, .4, 40)
ax.plot(xx, np.polyval(np.polyfit(oh[BRall], no[BRall], 1), xx), 'k--', lw=2.0,
        label=f'slope {sl:+.2f}')
for nm, mo, mn, col in (('N-poor', oh[A], no[A], C1), ('N-rich', oh[B], no[B], C2)):
    ax.plot(np.median(mo), np.median(mn), '*', color=col, ms=19, mec='w', mew=1.0,
            zorder=5, label=nm)
ax.plot(np.median(oh_g[GDEB]), np.median(no_g[GDEB]), 'D', color=cGSE, ms=9,
        mec='w', mew=1.0, zorder=5, label='GS/E debris')
ax.set(xlabel='[O/H]', ylabel='[N/O]', xlim=(-1.4, .4), ylim=(-.2, .4))
ax.legend(loc='upper left', fontsize=10.5)
tag(ax, '(f)  Q4: secondary nitrogen?', y=.60)

fig.tight_layout(pad=.6, w_pad=1.3, h_pad=1.1)
fig.savefig(f'{OUT}/au18_nfe_bimodality.png', bbox_inches='tight')
print(f'\nsaved {OUT}/au18_nfe_bimodality.png')
