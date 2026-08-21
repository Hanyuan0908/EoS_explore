"""Is the A-vs-B chemical difference just the birth-epoch radial gradient?

The two Eos channels (A = heated disc, B = born radial) differ in [Fe/H].  Stellar
abundances are frozen at birth, so the test is whether they also differ in *birth*
radius and birth time, and whether the [Fe/H] offset survives matching in (t_birth,
R_birth).  Reweights B onto A's (t_birth, R_birth) distribution and recomputes the
offset; a matched offset consistent with zero means the chemistry is the gradient.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import config_au18 as C
import channels_au18 as ch_mod

os.makedirs(C.FIG_DIR, exist_ok=True)
rng = np.random.default_rng(7)

d = ch_mod.load()
ids = d['ids']; eb = d['eps_birth']; e0 = d['eps_z0']; rz0 = d['r_z0']; feh = d['feh']
ELS = ch_mod.ELS; ratios = d['ratios']
Rb = d['R_birth']; zb = d['z_birth']; tb = d['tform']
CH_A, CH_B = d['A'], d['B']

finite = d['base']
print(f'merger-born with birth radius + chemistry: {finite.sum():,}')
print(f'cleaned channels: A={CH_A.sum():,}  B={CH_B.sum():,}')

SELECTIONS = {'all': finite,
              'r5_10': finite & (rz0 > 5) & (rz0 < 10),
              'r4_15': finite & (rz0 > 4) & (rz0 < 15)}


def channels(base):
    """Cleaned channels (shared definition) restricted to the given base sample."""
    return base & CH_A, base & CH_B


def weighted_median(x, w):
    o = np.argsort(x); x, w = x[o], w[o]
    c = np.cumsum(w) - .5 * w
    return np.interp(.5 * w.sum(), c, x)


REDGES = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 17, 25, 40, np.inf])
ZEDGES = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 9, 14, np.inf])


def matched_offset(A, B, quantity, dims=('t', 'R'), nboot=400):
    """Median(B) - median(A) after reweighting B onto A's birth-property grid.

    `dims` selects which birth coordinates to match on: t_birth, R_birth, |z_birth|.
    """
    axes_ = {'t': (tb, np.quantile(tb[A | B], np.linspace(0, 1, 6))),
             'R': (Rb, REDGES), 'z': (zb, ZEDGES)}
    cell = np.zeros(len(quantity), int); ncell = 1
    for d in dims:
        v, edges = axes_[d]
        i = np.clip(np.searchsorted(edges, v, 'right') - 1, 0, len(edges) - 2)
        cell = cell * (len(edges) - 1) + i; ncell *= len(edges) - 1
    nA = np.bincount(cell[A], minlength=ncell).astype(float)
    nB = np.bincount(cell[B], minlength=ncell).astype(float)
    shared = (nA > 0) & (nB > 0)
    # Weight each B star so B's occupancy of every shared cell equals A's.
    wB = np.where(shared[cell], nA[cell] / np.maximum(nB[cell], 1), 0.)[B]
    keepA = shared[cell][A]
    if wB.sum() == 0 or keepA.sum() == 0:
        return np.nan, np.nan, np.nan, 0., 0.
    xa, xb = quantity[A][keepA], quantity[B]
    raw = np.nanmedian(quantity[B]) - np.nanmedian(quantity[A])
    mat = weighted_median(xb, wB) - np.median(xa)
    boot = np.empty(nboot)
    for i in range(nboot):
        ia = rng.integers(0, len(xa), len(xa))
        ib = rng.integers(0, len(xb), len(xb))
        boot[i] = weighted_median(xb[ib], wB[ib]) - np.median(xa[ia])
    # Overlap fraction: how much of each channel lives in shared (t,R) cells.
    return raw, mat, boot.std(), keepA.mean(), (wB > 0).mean()


def gradient(mask, quantity):
    """Least-squares d(quantity)/dR over the well-populated birth-radius range."""
    q = mask & np.isfinite(quantity) & (Rb > 1) & (Rb < 20)
    return np.polyfit(Rb[q], quantity[q], 1)[0]


report = {}
for name, base in SELECTIONS.items():
    A, B = channels(base)
    print(f'\n=== {name}: base={base.sum():,}  A heated={A.sum():,}  B born-radial={B.sum():,} ===')
    for lab, m in [('A heated  ', A), ('B born-rad', B)]:
        print(f'  {lab} R_birth {np.percentile(Rb[m], [25, 50, 75]).round(2)}  '
              f'|z_birth| med={np.median(zb[m]):.2f}  '
              f'r_z0 {np.percentile(rz0[m], [25, 50, 75]).round(2)}  '
              f't_birth med={np.median(tb[m]):.2f}  [Fe/H] med={np.median(feh[m]):+.3f}')
    g = gradient(base, feh)
    dR = np.median(Rb[B]) - np.median(Rb[A])
    print(f'  birth [Fe/H] gradient (merger-born, 1<R<20) = {g:+.4f} dex/kpc')
    print(f'  Delta R_birth (B-A) = {dR:+.2f} kpc  ->  gradient predicts '
          f'Delta[Fe/H] = {g * dR:+.3f} dex')
    gz = np.polyfit(zb[base & (zb < 8)], feh[base & (zb < 8)], 1)[0]
    print(f'  birth [Fe/H] vertical gradient (|z|<8 kpc) = {gz:+.4f} dex/kpc; '
          f'median |z_birth| A={np.median(zb[A]):.2f} B={np.median(zb[B]):.2f} kpc')
    print(f'  {"quantity":9s} {"raw B-A":>9s} {"m(t,R)":>9s} {"+/-":>6s} {"m(t,R,z)":>9s} {"+/-":>6s} {"olapB":>6s}')
    report[name] = {}
    for q, y in [('Fe/H', feh)] + [(e + '/Fe', ratios[e]) for e in ELS]:
        raw, mat, err, oa, ob = matched_offset(A, B, y, dims=('t', 'R'))
        _, matz, errz, _, obz = matched_offset(A, B, y, dims=('t', 'R', 'z'))
        report[name][q] = (raw, mat, err, matz, errz)
        # With the |z_birth| cleaning the channels are disjoint in |z|, so no shared
        # cells exist and the z-matched estimate is undefined rather than null.
        if not np.isfinite(matz):
            zcol = f'{"n/a":>9s} {"":6s} {obz:6.2f}   <- disjoint in |z| by construction'
        else:
            zcol = (f'{matz:+9.3f} {errz:6.3f} {obz:6.2f}'
                    + ('' if abs(matz) > 2 * errz else '   <- consistent with 0'))
        flag = '' if abs(mat) > 2 * err else '   <- (t,R)-matched consistent with 0'
        print(f'  {q:9s} {raw:+9.3f} {mat:+9.3f} {err:6.3f} {zcol}{flag}')

np.savez(C.OUT_DIR + '/channel_radial_gradient.npz',
         **{f'{s}_{q.replace("/", "")}': np.asarray(v) for s, d in report.items() for q, v in d.items()})

# ---------------------------------------------------------------- figure ----
base = SELECTIONS['all']; A, B = channels(base)
fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
cA, cB = '#2166ac', '#b2182b'
LA, LB = f'A: heated disc (N={A.sum():,})', f'B: born radial (N={B.sum():,})'

ax = axes[0, 0]
bins = np.linspace(0, 30, 61)
for m, c, l in [(A, cA, LA), (B, cB, LB)]:
    ax.hist(Rb[m], bins=bins, density=True, histtype='step', lw=1.8, color=c, label=l)
    ax.axvline(np.median(Rb[m]), color=c, ls=':', lw=1.2)
ax.set(xlabel=r'$R_{\rm birth}$ [kpc]', ylabel='normalised', title='Birth radius')
ax.legend(fontsize=8)

ax = axes[0, 1]
bins = np.linspace(0, 30, 61)
for m, c in [(A, cA), (B, cB)]:
    ax.hist(rz0[m], bins=bins, density=True, histtype='step', lw=1.8, color=c)
    ax.axvline(np.median(rz0[m]), color=c, ls=':', lw=1.2)
ax.set(xlabel=r'$r_{z=0}$ [kpc]', ylabel='normalised', title='Present-day radius')

ax = axes[0, 2]
bins = np.linspace(np.floor(tb[base].min() * 10) / 10, np.ceil(tb[base].max() * 10) / 10, 41)
for m, c in [(A, cA), (B, cB)]:
    ax.hist(tb[m], bins=bins, density=True, histtype='step', lw=1.8, color=c)
ax.axvline(5.4, color='goldenrod', lw=2, alpha=.7, label='coalescence')
ax.set(xlabel='birth cosmic time [Gyr]', ylabel='normalised', title='Birth time')
ax.legend(fontsize=8)

# [Fe/H] vs birth radius: parent population in grey, channel running medians on top.
ax = axes[1, 0]
q = base & np.isfinite(feh)
ax.hexbin(Rb[q], feh[q], gridsize=60, extent=(0, 25, -2.5, .8), bins='log', mincnt=1, cmap='Greys')
redges = np.arange(0, 22, 1.5)
for m, c, l in [(A, cA, LA), (B, cB, LB)]:
    xs, ys = [], []
    for i in range(len(redges) - 1):
        w = m & (Rb >= redges[i]) & (Rb < redges[i + 1]) & np.isfinite(feh)
        if w.sum() > 15: xs.append(.5 * (redges[i] + redges[i + 1])); ys.append(np.median(feh[w]))
    ax.plot(xs, ys, color=c, lw=2, marker='o', ms=4, label=l)
g = gradient(base, feh)
ax.set(xlim=(0, 25), ylim=(-2.5, .8), xlabel=r'$R_{\rm birth}$ [kpc]', ylabel='[Fe/H]',
       title=f'Birth gradient {g:+.3f} dex/kpc')
ax.legend(fontsize=8, loc='lower left')

ax = axes[1, 1]
qs = list(report['all'].keys())
xpos = np.arange(len(qs))
raws = [report['all'][k][0] for k in qs]
mats = [report['all'][k][1] for k in qs]
errs = [report['all'][k][2] for k in qs]
matz = [report['all'][k][3] for k in qs]
errz = [report['all'][k][4] for k in qs]
ax.bar(xpos - .2, raws, .4, color='.6', label='raw B-A')
ax.bar(xpos + .2, mats, .4, yerr=errs, color=cB, label=r'matched $(t_{\rm b},R_{\rm b})$')
if np.all(np.isfinite(matz)):
    ax.bar(xpos + .27, matz, .27, yerr=errz, color='#f4a582',
           label=r'matched $(t_{\rm b},R_{\rm b},|z_{\rm b}|)$')
ax.axhline(0, color='k', lw=.8)
ax.set_xticks(xpos); ax.set_xticklabels(qs, rotation=45, ha='right')
ax.set(ylabel='B - A [dex]', title='Chemical offset before/after matching')
ax.legend(fontsize=8)

# Where each channel is born in the disc: R_birth vs |z_birth|.
ax = axes[1, 2]
for m, c, l in [(A, cA, LA), (B, cB, LB)]:
    ax.scatter(Rb[m], zb[m], s=3, c=c, alpha=.25, lw=0, label=l, rasterized=True)
gz = np.polyfit(zb[base & (zb < 8)], feh[base & (zb < 8)], 1)[0]
ax.set(xlim=(0, 25), ylim=(0, 12), xlabel=r'$R_{\rm birth}$ [kpc]',
       ylabel=r'$|z_{\rm birth}|$ [kpc]',
       title=f'Birth site geometry (vertical gradient {gz:+.3f} dex/kpc)')
ax.legend(fontsize=8, markerscale=3)

fig.suptitle('Au18 cleaned Eos channels: birth radius, present radius, and the birth-epoch '
             'metallicity gradient')
fig.tight_layout(rect=[0, 0, 1, .95])
out = C.FIG_DIR + '/au18_eos_channels_radial_gradient.png'
fig.savefig(out, dpi=150)
print('\nsaved', out)
