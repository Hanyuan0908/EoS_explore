"""Why sigma_[N/Fe] rises then falls WITHIN the bridge, at fixed birth site.

diag_nfe_dispersion_origin showed the dispersion is a property of the bridge:
halo-born stars born in the satellite or host apertures sit at ~0.016 dex at
every metallicity, bridge-born ones run 0.019 -> 0.057 -> 0.037.  That trend is
inside one site, so site cannot be its cause.  This script takes the bridge
halo-born stars apart per [Fe/H] bin and asks which of four things makes it.

  1  two components          a 2-Gaussian mixture is fitted by EM in each bin --
                             no divider assumed.  Separation Delta and weights w
                             come out of the fit, and the between-component
                             variance is w1 w2 Delta^2.
  2  the bin is not a point   [N/Fe] runs with [Fe/H], so a 0.2 dex bin converts
                             the local slope into spread: sigma_slope =
                             |d[N/Fe]/d[Fe/H]| x sd([Fe/H]) inside the bin.
  3  genuinely broad gas      what is left after removing 1 and 2: the residual
                             MAD about the within-bin linear trend, per component.
  4  small numbers            the bootstrap band, carried through.

The decomposition is sigma^2 = w1 w2 Delta^2 + sum_i w_i s_i^2, with the slope
term measured separately and compared against the total.  Whichever term traces
the observed shape is the answer, and the script prints all of them so the claim
can be checked rather than believed.

BIC compares the 2-component fit against 1 component in each bin, so "there are
two components here" is a measurement, not an assumption.

Writes figures/au18_nfe_bridge_trend.png
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
FBIN_W = 0.2
FBINS = np.arange(-1.2, 0.6 + 1e-9, FBIN_W)
NFIT, NBOOT = 60, 400
R_SAT, R_HOST = 6.0, 8.0
RNG = np.random.default_rng(0)
cBR, cSAT, cHOST = '#00897B', '#8E24AA', '#B8860B'
C1, C2 = '#4C72B0', '#C44E52'
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
ctr = .5 * (FBINS[:-1] + FBINS[1:])
NB = len(ctr)


def mad_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def em2(x, iters=500, tol=1e-10):
    """2-component 1D Gaussian mixture by EM.  Returns w, mu, s (sorted by mu), loglik."""
    x = np.asarray(x, float)
    mu = np.array([np.percentile(x, 20), np.percentile(x, 80)])
    s = np.full(2, max(np.std(x) / 2., 1e-3))
    w = np.array([.5, .5])
    ll_old = -np.inf
    for _ in range(iters):
        p = w[None, :] * np.exp(-.5 * ((x[:, None] - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))
        tot = np.maximum(p.sum(1), 1e-300)
        r = p / tot[:, None]
        ll = float(np.log(tot).sum())
        nk = np.maximum(r.sum(0), 1e-10)
        w = nk / len(x)
        mu = (r * x[:, None]).sum(0) / nk
        s = np.maximum(np.sqrt((r * (x[:, None] - mu) ** 2).sum(0) / nk), 1e-3)
        if abs(ll - ll_old) < tol * max(abs(ll), 1.):
            break
        ll_old = ll
    o = np.argsort(mu)
    return w[o], mu[o], s[o], ll


def ll1(x):
    m, sd = np.mean(x), max(np.std(x), 1e-3)
    return float((-.5 * ((x - m) / sd) ** 2 - np.log(sd * np.sqrt(2 * np.pi))).sum())


# ------------------------------- load (as before) ----------------------------
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
idx = o[p[ok]]
P, M = pos[idx], met[idx]
eps, zmx = eps_w[ok], zmx_w[ok]
disc = (eps > CUT) | (zmx < ZCUT)
del met, pos; gc.collect()

feh = C.bracket_abundance(M, 'Fe', 'H')
nfe = C.bracket_abundance(M, 'N', 'Fe')
halo = ~disc
d_gse = np.linalg.norm(P - g2, axis=1)
d_host = np.linalg.norm(P, axis=1)
sat = d_gse < R_SAT
host = (~sat) & (d_host < R_HOST)
bridge = ~(sat | host)
fin = np.isfinite(feh) & np.isfinite(nfe)
BR = bridge & halo & fin

# ------------------------------ the decomposition ----------------------------
cols = ['n', 'sig', 'std', 'siglo', 'sighi', 'w1', 'mu1', 's1', 'w2', 'mu2', 's2',
        'delta', 'between', 'within', 'slope', 'sigslope', 'resid', 'dbic']
R = {c: np.full(NB, np.nan) for c in cols}

for i in range(NB):
    b = BR & (feh >= FBINS[i]) & (feh < FBINS[i + 1])
    n = int(b.sum())
    R['n'][i] = n
    if n < NFIT:
        continue
    y, x = nfe[b], feh[b]
    R['sig'][i] = mad_sigma(y)
    # the mixture model below predicts a MOMENT sigma, so compare against std,
    # not against the robust MAD estimate which discounts a minority component
    R['std'][i] = float(np.std(y))
    bs = np.array([mad_sigma(y[RNG.integers(0, n, n)]) for _ in range(NBOOT)])
    R['siglo'][i], R['sighi'][i] = np.percentile(bs, (16, 84))

    w, mu, sd, ll2 = em2(y)
    R['w1'][i], R['mu1'][i], R['s1'][i] = w[0], mu[0], sd[0]
    R['w2'][i], R['mu2'][i], R['s2'][i] = w[1], mu[1], sd[1]
    R['delta'][i] = mu[1] - mu[0]
    R['between'][i] = np.sqrt(w[0] * w[1]) * (mu[1] - mu[0])
    R['within'][i] = np.sqrt((w * sd ** 2).sum())
    R['dbic'][i] = (2 * np.log(n) - 2 * ll1(y)) - (5 * np.log(n) - 2 * ll2)

    sl = np.polyfit(x, y, 1)[0]
    R['slope'][i] = sl
    R['sigslope'][i] = abs(sl) * np.std(x)
    R['resid'][i] = mad_sigma(y - np.polyval(np.polyfit(x, y, 1), x))

print(f'\nBRIDGE, halo-born, snapshot {SNAP} (t_form {t_lo:.2f}-{t_hi:.2f} Gyr)')
print(f'{"[Fe/H]":>7s} {"N":>6s} {"sigMAD":>7s} {"std":>7s} | {"w_low":>6s} {"mu_low":>7s} '
      f'{"s_low":>6s} {"mu_hi":>7s} {"s_hi":>6s} {"Delta":>7s} | '
      f'{"betwn":>6s} {"withn":>6s} {"model":>6s} | {"slope":>6s} {"sig_sl":>7s} '
      f'{"dBIC":>8s}')
for i in range(NB):
    if not np.isfinite(R['sig'][i]):
        continue
    mod = np.hypot(R['between'][i], R['within'][i])
    print(f"{ctr[i]:>+7.2f} {int(R['n'][i]):>6,} {R['sig'][i]:>7.4f} {R['std'][i]:>7.4f} | "
          f"{R['w1'][i]:>6.2f} {R['mu1'][i]:>+7.3f} {R['s1'][i]:>6.3f} "
          f"{R['mu2'][i]:>+7.3f} {R['s2'][i]:>6.3f} {R['delta'][i]:>7.3f} | "
          f"{R['between'][i]:>6.3f} {R['within'][i]:>6.3f} {mod:>6.3f} | "
          f"{R['slope'][i]:>+6.2f} {R['sigslope'][i]:>7.4f} {R['dbic'][i]:>8.0f}")

print('\ncontrols -- the same EM in the compact sites, where sigma is flat:')
for lab, m in (('satellite', sat & halo & fin), ('host', host & halo & fin)):
    print(f'  {lab}')
    for i in range(NB):
        b = m & (feh >= FBINS[i]) & (feh < FBINS[i + 1])
        if b.sum() < NFIT:
            continue
        w, mu, sd, ll2 = em2(nfe[b])
        db = (2 * np.log(b.sum()) - 2 * ll1(nfe[b])) - (5 * np.log(b.sum()) - 2 * ll2)
        print(f'    {ctr[i]:>+6.2f}  N={b.sum():>5,}  sigma={mad_sigma(nfe[b]):.4f}  '
              f'Delta={mu[1]-mu[0]:+.3f}  w_low={w[0]:.2f}  dBIC={db:>7.0f}')

# ---------------------------------- figure -----------------------------------
fig, AX = plt.subplots(3, 3, figsize=(16.5, 13.8))


def tag(ax, t, y=.96, **kw):
    ax.text(.03, y, t, transform=ax.transAxes, va='top', fontsize=12.5,
            fontweight='bold', **kw)


g = np.isfinite(R['sig'])
ax = AX[0, 0]
ax.plot(ctr[g], R['std'][g], color=cBR, lw=3.0, label=r'observed $\sigma$ (std)')
ax.plot(ctr[g], R['sig'][g], color=cBR, lw=1.8, ls=':', label=r'observed 1.48$\times$MAD')
ax.fill_between(ctr[g], R['siglo'][g], R['sighi'][g], color=cBR, alpha=.20, lw=0)
ax.plot(ctr[g], np.hypot(R['between'], R['within'])[g], 'k--', lw=2.0,
        label='2-component model')
ax.plot(ctr[g], R['between'][g], color=C2, lw=2.2, ls='-.',
        label=r'between: $\sqrt{w_1w_2}\,\Delta$')
ax.plot(ctr[g], R['within'][g], color=C1, lw=2.2, ls=':', label='within components')
ax.plot(ctr[g], R['sigslope'][g], color='0.45', lw=1.8,
        label=r'bin-width term $|{\rm d}[{\rm N/Fe}]/{\rm d[Fe/H]}|\,{\rm sd}$')
ax.set(xlabel='[Fe/H]', ylabel=r'$\sigma_{\rm [N/Fe]}$ [dex]', xlim=(-1.2, .6))
ax.legend(loc='upper left', fontsize=10, handlelength=2.0)
tag(ax, '(a)  what makes the bridge trend', y=.66)

ax = AX[0, 1]
ax.plot(ctr[g], R['mu1'][g], 'o-', color=C1, lw=2.2, ms=5, label=r'N-poor $\mu$')
ax.plot(ctr[g], R['mu2'][g], 'o-', color=C2, lw=2.2, ms=5, label=r'N-rich $\mu$')
ax.fill_between(ctr[g], R['mu1'][g], R['mu2'][g], color='0.6', alpha=.22, lw=0)
for i in np.flatnonzero(g):
    ax.errorbar(ctr[i], R['mu1'][i], yerr=R['s1'][i], color=C1, capsize=3, lw=1.2)
    ax.errorbar(ctr[i], R['mu2'][i], yerr=R['s2'][i], color=C2, capsize=3, lw=1.2)
ax.set(xlabel='[Fe/H]', ylabel=r'[N/Fe] of each component', xlim=(-1.2, .6))
ax.legend(loc='lower right', fontsize=10.5)
tag(ax, '(b)  the components converge at low [Fe/H]')

ax = AX[0, 2]
ax.plot(ctr[g], R['delta'][g], 'o-', color='k', lw=2.4, ms=5, label=r'$\Delta$ [dex]')
ax2 = ax.twinx()
ax2.plot(ctr[g], (R['w1'] * R['w2'])[g], 's--', color=cBR, lw=2.2, ms=5,
         label=r'$w_1w_2$')
ax2.set_ylabel(r'$w_1w_2$', color=cBR); ax2.tick_params(axis='y', colors=cBR)
ax2.set_ylim(0, .28)
ax.set(xlabel='[Fe/H]', ylabel=r'component separation $\Delta$ [dex]',
       xlim=(-1.2, .6), ylim=(0, None))
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=10.5)
tag(ax, '(c)  separation drives it, not the mix', y=.80)

ax = AX[1, 0]
ax.plot(ctr[g], R['sig'][g], color=cBR, lw=2.8, label=r'$\sigma$, raw')
ax.plot(ctr[g], R['resid'][g], color='0.25', lw=2.4, ls='--',
        label=r'$\sigma$ after removing the within-bin slope')
ax.plot(ctr[g], R['within'][g], color=C1, lw=2.2, ls=':',
        label='within components')
ax.set(xlabel='[Fe/H]', ylabel=r'$\sigma_{\rm [N/Fe]}$ [dex]', xlim=(-1.2, .6),
       ylim=(0, None))
ax.legend(loc='upper left', fontsize=10.5)
tag(ax, '(d)  the bin-width term is not the cause', y=.70)

ax = AX[1, 1]
ax.plot(ctr[g], R['dbic'][g], 'o-', color='k', lw=2.2, ms=5)
ax.axhline(0, color='0.5', lw=1.0)
ax.axhline(10, color=C2, lw=1.2, ls='--')
ax.text(.97, .06, r'$\Delta$BIC = 10: strong evidence for 2 components',
        transform=ax.transAxes, ha='right', fontsize=10.5, color=C2)
ax.set(xlabel='[Fe/H]', ylabel=r'$\Delta$BIC (1 comp $-$ 2 comp)', xlim=(-1.2, .6))
ax.set_yscale('symlog', linthresh=10)
tag(ax, '(e)  is there really more than one component?')

ax = AX[1, 2]
ax.plot(ctr[g], R['slope'][g], 'o-', color='0.25', lw=2.4, ms=5,
        label='bridge')
for lab, m, col in (('satellite', sat & halo & fin, cSAT),
                    ('host', host & halo & fin, cHOST)):
    sl = np.full(NB, np.nan)
    for i in range(NB):
        b = m & (feh >= FBINS[i]) & (feh < FBINS[i + 1])
        if b.sum() >= NFIT:
            sl[i] = np.polyfit(feh[b], nfe[b], 1)[0]
    gg = np.isfinite(sl)
    ax.plot(ctr[gg], sl[gg], 'o--', color=col, lw=2.0, ms=4, label=lab)
ax.axhline(0, color='0.6', lw=1.0)
ax.set(xlabel='[Fe/H]', ylabel=r'${\rm d}[{\rm N/Fe}]/{\rm d[Fe/H]}$ in the bin',
       xlim=(-1.2, .6))
ax.legend(loc='upper right', fontsize=10.5)
tag(ax, '(f)  the local N-Fe slope')


# ---- bottom row: what Delta actually is, drawn ------------------------------
# Delta is the gap between the means of the two Gaussians fitted to the [N/Fe]
# values of the bridge stars in one [Fe/H] bin.  Nothing more.  Three bins are
# shown: below the peak, at it, and above it.
PICK = [i for i in range(NB) if np.isfinite(R['sig'][i])]
PICK = [PICK[1], PICK[4], PICK[-1]]
xx = np.linspace(-.10, .28, 400)
for j, i in enumerate(PICK):
    ax = AX[2, j]
    b = BR & (feh >= FBINS[i]) & (feh < FBINS[i + 1])
    y = nfe[b]
    ax.hist(y, bins=np.arange(-.10, .281, .012), density=True,
            histtype='stepfilled', color=cBR, alpha=.25, lw=0)
    ax.hist(y, bins=np.arange(-.10, .281, .012), density=True,
            histtype='step', color=cBR, lw=2.0, label=f'bridge stars ({b.sum():,})')
    w = np.array([R['w1'][i], R['w2'][i]])
    mu = np.array([R['mu1'][i], R['mu2'][i]])
    sd = np.array([R['s1'][i], R['s2'][i]])
    tot = np.zeros_like(xx)
    for q, (col, nm) in enumerate(((C1, 'N-poor'), (C2, 'N-rich'))):
        gcurve = w[q] * np.exp(-.5 * ((xx - mu[q]) / sd[q]) ** 2) / (sd[q] * np.sqrt(2 * np.pi))
        tot += gcurve
        ax.plot(xx, gcurve, color=col, lw=2.2,
                label=f'{nm}: $\mu$={mu[q]:+.3f}, $w$={w[q]:.2f}')
        ax.axvline(mu[q], color=col, lw=1.0, ls=':')
    ax.plot(xx, tot, 'k--', lw=1.6, label='sum of the two')
    ytop = ax.get_ylim()[1]
    ax.annotate('', xy=(mu[1], .80 * ytop), xytext=(mu[0], .80 * ytop),
                arrowprops=dict(arrowstyle='<->', color='k', lw=1.8))
    ax.text(.5 * (mu[0] + mu[1]), .83 * ytop,
            f'$\Delta$ = {mu[1] - mu[0]:.3f} dex', ha='center', fontsize=12.5,
            bbox=dict(fc='white', ec='none', alpha=.85, pad=1.5))
    ax.set(xlabel='[N/Fe]', ylabel='density' if j == 0 else '', xlim=(-.10, .28),
           title=f'[Fe/H] = {FBINS[i]:+.1f} to {FBINS[i+1]:+.1f}'
                 f'   ($\sigma$ = {R["std"][i]:.3f})')
    ax.set_ylim(top=ytop * 1.30)
    ax.legend(loc='upper left', fontsize=9.5)
    ax.text(.97, .99, f'({"ghi"[j]})  $\Delta$ is this gap', transform=ax.transAxes,
            va='top', ha='right', fontsize=12.5, fontweight='bold')

fig.tight_layout(pad=.6, w_pad=1.3, h_pad=1.3)
fig.savefig(f'{OUT}/au18_nfe_bridge_trend.png', bbox_inches='tight')
print(f'\nsaved {OUT}/au18_nfe_bridge_trend.png')
