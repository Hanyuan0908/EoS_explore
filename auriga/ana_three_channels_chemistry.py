"""Chemistry of the three Eos-relevant Au18 populations.

  A heated disc  - born in the disc during the merger, hot at z=0
  B born radial  - born hot off-plane during the merger        (Eos analogue)
  C splash       - born in the disc *before* the merger, hot at z=0 (Splash analogue)

Au18 has almost no alpha spread (see printout), so the comparison is driven by
[Fe/H].  Offsets are therefore also reported after reweighting each population
onto A's [Fe/H] distribution, which is the only meaningful chemical control here.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import config_au18 as C
import channels_au18 as ch_mod

os.makedirs(C.FIG_DIR, exist_ok=True)
rng = np.random.default_rng(11)
ELS = ch_mod.ELS

d = ch_mod.load()
sp = np.load(C.OUT_DIR + '/premerger_splash.npz')
good = np.isfinite(sp['eps_birth']) & np.isfinite(sp['eps_z0']) & np.isfinite(sp['z_birth'])
Csel = good & (sp['eps_birth'] > .7) & (sp['z_birth'] < ch_mod.Z_A_MAX) & (sp['eps_z0'] < .3)

POPS = {
    'A heated disc': dict(feh=d['feh'][d['A']], t=d['tform'][d['A']], z=d['z_birth'][d['A']],
                          R=d['R_birth'][d['A']], r0=d['r_z0'][d['A']], c='#2166ac',
                          ratios={e: d['ratios'][e][d['A']] for e in ELS}),
    'B born radial': dict(feh=d['feh'][d['B']], t=d['tform'][d['B']], z=d['z_birth'][d['B']],
                          R=d['R_birth'][d['B']], r0=d['r_z0'][d['B']], c='#7b3294',
                          ratios={e: d['ratios'][e][d['B']] for e in ELS}),
    'C splash': dict(feh=sp['feh'][Csel], t=sp['tform'][Csel], z=sp['z_birth'][Csel],
                     R=sp['R_birth'][Csel], r0=sp['r_z0'][Csel], c='#e08214',
                     ratios={e: sp[e.lower() + 'fe'][Csel] for e in ELS}),
}
for k, v in POPS.items(): v['n'] = len(v['feh'])
print('population        N     t_birth  R_birth  |z_b|   r_z0    [Fe/H]')
for k, v in POPS.items():
    print(f'{k:16s} {v["n"]:6,d} {np.median(v["t"]):8.2f} {np.median(v["R"]):8.2f} '
          f'{np.median(v["z"]):6.2f} {np.median(v["r0"]):6.2f} {np.median(v["feh"]):+8.3f}')

print('\nalpha spread within each population (5-95 pct width, dex):')
for k, v in POPS.items():
    w = {e: np.diff(np.percentile(v['ratios'][e], [5, 95]))[0] for e in ['O', 'Mg', 'Si']}
    print(f'  {k:16s} ' + '  '.join(f'{e}:{w[e]:.3f}' for e in w)
          + f'   [Fe/H]:{np.diff(np.percentile(v["feh"], [5, 95]))[0]:.3f}')


def weighted_median(x, w):
    o = np.argsort(x); x, w = x[o], w[o]
    c = np.cumsum(w) - .5 * w
    return np.interp(.5 * w.sum(), c, x)


def feh_matched(ref, tgt, quantity, nboot=400):
    """Median(tgt) - median(ref) after reweighting tgt onto ref's [Fe/H] histogram."""
    edges = np.arange(-2.5, 1.01, .1)
    ir = np.clip(np.searchsorted(edges, ref['feh'], 'right') - 1, 0, len(edges) - 2)
    it = np.clip(np.searchsorted(edges, tgt['feh'], 'right') - 1, 0, len(edges) - 2)
    nr = np.bincount(ir, minlength=len(edges) - 1).astype(float)
    nt = np.bincount(it, minlength=len(edges) - 1).astype(float)
    shared = (nr > 0) & (nt > 0)
    w = np.where(shared[it], nr[it] / np.maximum(nt[it], 1), 0.)
    keep = shared[ir]
    if w.sum() == 0 or keep.sum() == 0: return np.nan, np.nan, 0.
    xr = quantity[0][keep]; xt = quantity[1]
    mat = weighted_median(xt, w) - np.median(xr)
    boot = np.empty(nboot)
    for i in range(nboot):
        a = rng.integers(0, len(xr), len(xr)); b = rng.integers(0, len(xt), len(xt))
        boot[i] = weighted_median(xt[b], w[b]) - np.median(xr[a])
    return mat, boot.std(), (w > 0).mean()


print('\nchemical offsets relative to A (raw / [Fe/H]-matched):')
print(f'  {"pair":18s} {"quantity":9s} {"raw":>8s} {"matched":>9s} {"+/-":>6s} {"olap":>6s}')
offsets = {}
for name in ['B born radial', 'C splash']:
    ref, tgt = POPS['A heated disc'], POPS[name]
    offsets[name] = {}
    for q in ['Fe/H'] + [e + '/Fe' for e in ELS]:
        yr = ref['feh'] if q == 'Fe/H' else ref['ratios'][q[:-3]]
        yt = tgt['feh'] if q == 'Fe/H' else tgt['ratios'][q[:-3]]
        raw = np.median(yt) - np.median(yr)
        mat, err, ol = feh_matched(ref, tgt, (yr, yt))
        offsets[name][q] = (raw, mat, err)
        print(f'  {name+" - A":18s} {q:9s} {raw:+8.3f} {mat:+9.3f} {err:6.3f} {ol:6.2f}')

# The science question: is C (Splash) separable from B (Eos) at fixed [Fe/H]?
print('\nC splash vs B born radial:')
ref, tgt = POPS['B born radial'], POPS['C splash']
for q in ['Fe/H'] + [e + '/Fe' for e in ELS]:
    yr = ref['feh'] if q == 'Fe/H' else ref['ratios'][q[:-3]]
    yt = tgt['feh'] if q == 'Fe/H' else tgt['ratios'][q[:-3]]
    mat, err, ol = feh_matched(ref, tgt, (yr, yt))
    print(f'  {"C - B":18s} {q:9s} {np.median(yt)-np.median(yr):+8.3f} {mat:+9.3f} {err:6.3f} {ol:6.2f}')

# ------------------------------------------------------------------ figure --
fig = plt.figure(figsize=(18, 10.5))
gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.05], hspace=.42, wspace=.35)

for col, (key, lab, rng_) in enumerate([('feh', '[Fe/H]', (-2.2, .8)),
                                        ('t', 'birth cosmic time [Gyr]', (3.2, 6.7)),
                                        ('z', r'$|z_{\rm birth}|$ [kpc]', (0, 12)),
                                        ('R', r'$R_{\rm birth}$ [kpc]', (0, 25)),
                                        ('r0', r'$r_{z=0}$ [kpc]', (0, 25))]):
    ax = fig.add_subplot(gs[0, col])
    bins = np.linspace(*rng_, 46)
    for k, v in POPS.items():
        ax.hist(v[key], bins=bins, density=True, histtype='step', lw=1.9, color=v['c'],
                label=f'{k} (N={v["n"]:,})')
    ax.set(xlabel=lab, ylabel='normalised' if col == 0 else '')
    if col == 0: ax.legend(fontsize=7.5)
ax = fig.add_subplot(gs[0, 5])
ax.axvline(5.4, color='goldenrod', lw=2, alpha=.8)
for k, v in POPS.items():
    xs = np.sort(v['t']); ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=v['c'], lw=1.9)
ax.set(xlabel='birth time [Gyr]', ylabel='cumulative', title='coalescence = gold', ylim=(0, 1))

# Running medians of [X/Fe] against [Fe/H]: the alpha tracks of the three populations.
fedges = np.arange(-2.2, .81, .2)
fc = .5 * (fedges[:-1] + fedges[1:])
for j, el in enumerate(ELS):
    ax = fig.add_subplot(gs[1, j])
    for k, v in POPS.items():
        y = v['ratios'][el]; xs, ys, lo, hi = [], [], [], []
        for i in range(len(fedges) - 1):
            m = (v['feh'] >= fedges[i]) & (v['feh'] < fedges[i + 1]) & np.isfinite(y)
            if m.sum() > 25:
                xs.append(fc[i]); ys.append(np.median(y[m]))
                lo.append(np.percentile(y[m], 25)); hi.append(np.percentile(y[m], 75))
        ax.plot(xs, ys, color=v['c'], lw=2, marker='o', ms=3)
        ax.fill_between(xs, lo, hi, color=v['c'], alpha=.15, lw=0)
    ax.set(xlabel='[Fe/H]', ylabel=f'[{el}/Fe]' if j == 0 else '', title=f'[{el}/Fe]')
    ax.set_xlim(-2.2, .8)

ax = fig.add_subplot(gs[2, :3])
qs = ['Fe/H'] + [e + '/Fe' for e in ELS]
xp = np.arange(len(qs))
for off, (name, mk) in enumerate(zip(['B born radial', 'C splash'], [-.2, .2])):
    ax.bar(xp + mk, [offsets[name][q][1] for q in qs], .4,
           yerr=[offsets[name][q][2] for q in qs], color=POPS[name]['c'], label=f'{name} - A')
ax.axhline(0, color='k', lw=.8)
ax.set_xticks(xp); ax.set_xticklabels(qs, rotation=45, ha='right')
ax.set(ylabel='offset from A [dex]', title='Offsets from A after matching in [Fe/H]')
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[2, 3:])
for k, v in POPS.items():
    ax.scatter(v['R'], v['z'], s=3, c=v['c'], alpha=.22, lw=0, rasterized=True, label=k)
ax.set(xlim=(0, 25), ylim=(0, 12), xlabel=r'$R_{\rm birth}$ [kpc]',
       ylabel=r'$|z_{\rm birth}|$ [kpc]', title='Birth sites')
ax.legend(fontsize=8, markerscale=3)

fig.suptitle('Au18: chemistry of the three Eos-relevant populations '
             '(A merger-born disc, B merger-born radial, C pre-merger Splash)', fontsize=13)
out = C.FIG_DIR + '/au18_three_channels_chemistry.png'
fig.savefig(out, dpi=140, bbox_inches='tight')
np.savez(C.OUT_DIR + '/three_channels.npz', C_ids=sp['ids'][Csel],
         **{f'{k.split()[0]}_{q.replace("/", "")}': np.asarray(v)
            for k, dd in offsets.items() for q, v in dd.items()})
print('\nsaved', out)
