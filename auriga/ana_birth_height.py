"""Birth-height distributions of the two Au18 Eos channels.

|z_birth| turns out to separate the heated-disc channel (A) from the merger-induced
born-radial channel (B) far better than the birth chemistry does, so this quantifies
it: distributions, effective scale heights, and the separation power of a |z| cut.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, mannwhitneyu
import config_au18 as C

os.makedirs(C.FIG_DIR, exist_ok=True)

ch = np.load(C.OUT_DIR + '/eos_two_channels.npz')
br = np.load(C.OUT_DIR + '/merger_birth_radii.npz')
o = np.argsort(br['ids']); bids = br['ids'][o]
p = np.searchsorted(bids, ch['ids'])
ok = (p < len(bids)) & (bids[np.minimum(p, len(bids) - 1)] == ch['ids'])
ix = o[p[ok]]
eb = ch['eps_birth'][ok]; e0 = ch['eps_z0'][ok]; rz0 = ch['r_z0'][ok]; feh = ch['feh'][ok]
Rb = br['R_birth'][ix]; zb = br['z_birth'][ix]; tb = br['tform'][ix]

base = np.isfinite(eb) & np.isfinite(e0) & np.isfinite(zb)
A = base & (eb > .7) & (e0 < .3)      # born cold -> hot now: heated disc
B = base & (eb < .3) & (e0 < .3)      # born hot  -> hot now: born radial
cA, cB, cP = '#2166ac', '#b2182b', '.55'
LA = f'A: heated disc (N={A.sum():,})'
LB = f'B: born radial (N={B.sum():,})'
LP = f'all merger-born (N={base.sum():,})'

# Effective exponential scale height: for an exponential, median = h ln2.
print('quantity                    A heated   B born-radial   all merger-born')
for lab, f in [('median |z_birth| [kpc]', np.median),
               ('  90th pct [kpc]     ', lambda x: np.percentile(x, 90))]:
    print(f'{lab:26s} {f(zb[A]):9.2f} {f(zb[B]):14.2f} {f(zb[base]):17.2f}')
print(f'{"eff. scale height [kpc]":26s} {np.median(zb[A])/np.log(2):9.2f} '
      f'{np.median(zb[B])/np.log(2):14.2f} {np.median(zb[base])/np.log(2):17.2f}')
for lab, thr in [('frac |z_b| < 1 kpc', 1.), ('frac |z_b| > 3 kpc', 3.)]:
    fa = (zb[A] < thr).mean() if '<' in lab else (zb[A] > thr).mean()
    fb = (zb[B] < thr).mean() if '<' in lab else (zb[B] > thr).mean()
    fp = (zb[base] < thr).mean() if '<' in lab else (zb[base] > thr).mean()
    print(f'{lab:26s} {fa:9.3f} {fb:14.3f} {fp:17.3f}')

ks = ks_2samp(zb[A], zb[B])
auc = mannwhitneyu(zb[B], zb[A]).statistic / (A.sum() * B.sum())
print(f'\nKS(|z_birth| A vs B): D={ks.statistic:.3f}  p={ks.pvalue:.2g}')
print(f'AUC (P[|z_b| of B > |z_b| of A]) = {auc:.3f}')
# Compare against how well the birth chemistry separates them.
auc_feh = mannwhitneyu(feh[A], feh[B]).statistic / (A.sum() * B.sum())
auc_R = mannwhitneyu(Rb[B], Rb[A]).statistic / (A.sum() * B.sum())
print(f'  for reference: AUC([Fe/H]) = {auc_feh:.3f},  AUC(R_birth) = {auc_R:.3f}')

# Separation power of a simple |z_birth| threshold.
cuts = np.arange(.25, 8.01, .25)
compB = np.array([(zb[B] > c).mean() for c in cuts])          # B kept
contA = np.array([(zb[A] > c).mean() for c in cuts])          # A leaking through
purity = np.array([(zb[B] > c).sum() / max((zb[B] > c).sum() + (zb[A] > c).sum(), 1)
                   for c in cuts])
prior = B.sum() / (A.sum() + B.sum())
j = np.argmax(compB - contA)
print(f'\nbest |z_birth| cut (max completeness-contamination): {cuts[j]:.2f} kpc  ->  '
      f'keeps {compB[j]:.1%} of B, {contA[j]:.1%} of A; purity {purity[j]:.1%} '
      f'(baseline {prior:.1%})')

# ------------------------------------------------------------------ figure --
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.6))

ax = axes[0, 0]
bins = np.linspace(0, 12, 61)
ax.hist(zb[base], bins=bins, density=True, histtype='stepfilled', color=cP, alpha=.28, label=LP)
for m, c, l in [(A, cA, LA), (B, cB, LB)]:
    ax.hist(zb[m], bins=bins, density=True, histtype='step', lw=2, color=c, label=l)
    ax.axvline(np.median(zb[m]), color=c, ls=':', lw=1.4)
ax.set(xlabel=r'$|z_{\rm birth}|$ [kpc]', ylabel='normalised density', xlim=(0, 12),
       title='Birth height (dotted = median)')
ax.legend(fontsize=8.5)

ax = axes[0, 1]
ax.hist(zb[base], bins=bins, density=True, histtype='stepfilled', color=cP, alpha=.28)
for m, c in [(A, cA), (B, cB)]:
    ax.hist(zb[m], bins=bins, density=True, histtype='step', lw=2, color=c)
ax.set_yscale('log')
ax.set(xlabel=r'$|z_{\rm birth}|$ [kpc]', ylabel='normalised density (log)', xlim=(0, 12),
       title=f'Same, log scale — eff. scale height '
             f'{np.median(zb[A])/np.log(2):.1f} vs {np.median(zb[B])/np.log(2):.1f} kpc')

ax = axes[1, 0]
for m, c, l in [(base, cP, LP), (A, cA, LA), (B, cB, LB)]:
    xs = np.sort(zb[m]); ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=c, lw=2, label=l)
ax.axhline(.5, color='k', lw=.6, ls='--')
ax.set(xlabel=r'$|z_{\rm birth}|$ [kpc]', ylabel='cumulative fraction', xlim=(0, 12), ylim=(0, 1),
       title=f'Cumulative: KS D={ks.statistic:.2f}, AUC={auc:.2f}')
ax.legend(fontsize=8.5, loc='lower right')

# Does the separation hold at every birth time / birth radius, or is it an average?
ax = axes[1, 1]
tedges = np.linspace(tb[base].min(), tb[base].max(), 9)
for m, c, l in [(A, cA, LA), (B, cB, LB)]:
    xs, ys, lo, hi = [], [], [], []
    for i in range(len(tedges) - 1):
        w = m & (tb >= tedges[i]) & (tb < tedges[i + 1])
        if w.sum() > 20:
            xs.append(.5 * (tedges[i] + tedges[i + 1])); ys.append(np.median(zb[w]))
            lo.append(np.percentile(zb[w], 25)); hi.append(np.percentile(zb[w], 75))
    ax.plot(xs, ys, color=c, lw=2, marker='o', ms=4, label=l)
    ax.fill_between(xs, lo, hi, color=c, alpha=.16, lw=0)
ax.axvline(5.4, color='goldenrod', lw=2, alpha=.75, label='coalescence')
ax.set(xlabel='birth cosmic time [Gyr]', ylabel=r'median $|z_{\rm birth}|$ [kpc]',
       title='Birth height vs birth time (band = 25-75 pct)')
ax.legend(fontsize=8.5)

fig.suptitle('Au18: birth height separates the heated-disc and merger-induced Eos channels')
fig.tight_layout(rect=[0, 0, 1, .95])
out = C.FIG_DIR + '/au18_eos_channels_birth_height.png'
fig.savefig(out, dpi=150)
np.savez(C.OUT_DIR + '/eos_channels_birth_height.npz',
         z_birth_A=zb[A], z_birth_B=zb[B], z_birth_all=zb[base],
         cuts=cuts, completeness_B=compB, contamination_A=contA, purity=purity,
         ks_D=ks.statistic, auc=auc)
print('\nsaved', out)
