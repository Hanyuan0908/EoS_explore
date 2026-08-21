"""Reproduction of Figure 5 of Borbolato et al. (2026) for the Clumpy+merger model.

Their Figure 5 has four columns -- Isolated Clumpy (1-2) and Clumpy+merger (3-4),
each split into low- and high-alpha Splash.  Only the merger runs are available
here, so this reproduces the Clumpy+merger half:

  top row     t_form vs R_form, coloured by [Fe/H], with the dwarf's pericentric
              passages at t = 1.6, 2.5 and 3.2 Gyr marked;
  bottom row  the evolution of median V_phi for the Splash populations against
              their canonical discs, same pericentre marks, grey band = the
              clumpy phase (first 3 Gyr).

Two extra panels carry the selection provenance, since the alpha split is the
step that has to be inferred rather than read off: the [O/Fe]-[Fe/H] plane with
the boundary drawn, and the [O/Fe] histogram in their -0.7 < [Fe/H] < -0.2 window
with the valley and the exclusion gap marked.

Reads out/fig5_clumpy_merger.npz (built by gastro_fig5_prep.py).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gastro_config as G

os.makedirs(G.FIG_DIR, exist_ok=True)
d = np.load(G.OUT_DIR + '/fig5_clumpy_merger.npz')

PERI = (1.6, 2.5, 3.2)            # first, second, final pericentric passage (their Sec. 2.2)
CLUMPY_PHASE = (0., 3.)
C_LOW, C_HIGH, C_DISC = '#1a9850', '#e08214', '0.15'
FEH_WINDOW = (-0.7, -0.2)

ofe, feh = d['ofe'], d['feh']
tform, Rform, vphi0 = d['tform'], d['Rform'], d['vphi']
dip, gap, RMIN, Rd = float(d['dip']), float(d['gap']), float(d['RMIN']), float(d['Rd'])
times, counts = d['times'], d['counts']

# The selection is rebuilt here rather than read from the prep file: the cached
# per-snapshot kinematics make every cut a plotting-time decision, so the alpha
# split or the satellite definition can be revised without re-walking the series.
MODEL_DIR = G.HERE + '/jrun003.dwarfM06XY138Z37Vxy20FB20'
NAME = 'dwarfM06XY138Z37Vxy20FB20'
zform = np.load(f'{MODEL_DIR}/{NAME}_zform.npy')
insitu = ~G.satellite_born(Rform, zform)
vol = insitu & (d['R'] > RMIN)
low = vol & (ofe < dip - gap)
high = vol & (ofe > dip + gap)
sl = low & (vphi0 < G.EOS_VPHI_MAX)
sh = high & (vphi0 < 50.)
print(f'satellite-born excluded: {(~insitu).sum():,} stars')

print(f'R_d={Rd:.2f} kpc -> R>{RMIN:.2f} kpc;  alpha split at [O/Fe]={dip:+.3f} +/-{gap}')
print(f'low-alpha {low.sum():,} -> Splash {sl.sum():,} ({100*sl.sum()/low.sum():.2f}%)')
print(f'high-alpha {high.sum():,} -> Splash {sh.sum():,} ({100*sh.sum()/high.sum():.2f}%)')


def track(mask, stat=np.median):
    """Reconstruct a population's V_phi history from the cached per-snapshot arrays."""
    off = np.concatenate([[0], np.cumsum(counts)])
    out = np.full(len(times), np.nan)
    for k, n in enumerate(counts):
        v = d['snap_vphi'][off[k]:off[k] + n][mask[:n]]
        if len(v) > 50:
            out[k] = stat(v)
    return out


fig, axes = plt.subplots(2, 3, figsize=(19, 10.2))

# --- (a) the alpha plane, with the boundary that defines everything else ------
ax = axes[0, 0]
ax.hist2d(feh[vol], ofe[vol], bins=(150, 130), range=((-2.0, 0.8), (-0.45, 0.6)),
          norm=LogNorm(), cmap='Greys')
ax.axhspan(dip - gap, dip + gap, color='r', alpha=.25, lw=0)
ax.axhline(dip, color='r', lw=1.6)
for v in FEH_WINDOW:
    ax.axvline(v, color='tab:blue', lw=1.2, ls='--')
ax.set(xlabel='[Fe/H]', ylabel='[O/Fe]',
       title=f'(a) $\\alpha$ split at [O/Fe]$={dip:+.2f}$ (red), $\\pm${gap} gap\n'
             f'dashed = the ${FEH_WINDOW[0]}<$[Fe/H]$<{FEH_WINDOW[1]}$ window')

# --- (d) the histogram the split is read from --------------------------------
ax = axes[1, 0]
ax.plot(d['ofe_hist_x'], d['ofe_hist_y'], color='k', lw=1.8)
ax.axvspan(dip - gap, dip + gap, color='r', alpha=.25, lw=0)
ax.axvline(dip, color='r', lw=1.6)
ax.set(xlim=(-0.45, 0.6), xlabel='[O/Fe]', ylabel='stars per bin',
       title=f'(d) [O/Fe] within ${FEH_WINDOW[0]}<$[Fe/H]$<{FEH_WINDOW[1]}$\n'
             'split = deepest valley, not a peak-first guess')
ax.text(dip - gap - .02, ax.get_ylim()[1] * .9, 'low-$\\alpha$', ha='right', color=C_LOW, fontsize=11)
ax.text(dip + gap + .02, ax.get_ylim()[1] * .9, 'high-$\\alpha$', ha='left', color=C_HIGH, fontsize=11)

# --- (b), (c) t_form vs R_form ------------------------------------------------
for col, (m, c, name, n_all) in enumerate(
        [(sl, C_LOW, f'low-$\\alpha$ Splash ($V_\\phi<{G.EOS_VPHI_MAX:.0f}$)', low.sum()),
         (sh, C_HIGH, f'high-$\\alpha$ Splash ($V_\\phi<50$)', high.sum())], start=1):
    ax = axes[0, col]
    sc = ax.scatter(Rform[m], tform[m], c=feh[m], s=3, alpha=.5, lw=0,
                    cmap='viridis', vmin=-1.2, vmax=0.3)
    for p in PERI:
        ax.axhline(p, color='k', lw=1.1, ls='--')
    ax.axhspan(*CLUMPY_PHASE, color='0.5', alpha=.18, lw=0)
    ax.set(xlim=(0, 20), ylim=(0, 10), xlabel=r'$R_{\rm form}$ [kpc]',
           ylabel='$t_{\\rm form}$ [Gyr]' if col == 1 else '',
           title=f'({"bc"[col-1]}) {name}\nN={m.sum():,} of {n_all:,} '
                 f'({100*m.sum()/n_all:.1f}%)')
    plt.colorbar(sc, ax=ax, label='[Fe/H]', pad=.01)

# --- (e) V_phi evolution ------------------------------------------------------
ax = axes[1, 1]
for m, c, lab, ls in [(sl, C_LOW, 'low-$\\alpha$ Splash', '-'),
                      (sh, C_HIGH, 'high-$\\alpha$ Splash', '-'),
                      (low & ~sl, C_DISC, 'canonical low-$\\alpha$ disc', '-'),
                      (high & ~sh, C_DISC, 'canonical high-$\\alpha$ disc', '--')]:
    ax.plot(times, track(m), color=c, ls=ls, lw=2.2, marker='o', ms=3.5, label=lab)
for p in PERI:
    ax.axvline(p, color='k', lw=1.1, ls='--')
ax.axvspan(*CLUMPY_PHASE, color='0.5', alpha=.18, lw=0, label='clumpy phase')
ax.set(xlabel='time [Gyr]', ylabel=r'median $V_\phi$ [km s$^{-1}$]',
       title='(e) $V_\\phi$ evolution of the Splash vs its parent disc')
ax.legend(fontsize=8.5, loc='center right')

# --- (f) where the Splash cuts fall at z=0 -----------------------------------
ax = axes[1, 2]
b = np.linspace(-250, 400, 80)
ax.hist(vphi0[low], bins=b, density=True, histtype='step', lw=2, color=C_LOW,
        label=f'low-$\\alpha$ (N={low.sum():,})')
ax.hist(vphi0[high], bins=b, density=True, histtype='step', lw=2, color=C_HIGH,
        label=f'high-$\\alpha$ (N={high.sum():,})')
ax.axvline(G.EOS_VPHI_MAX, color=C_LOW, lw=1.6, ls='--')
ax.axvline(50, color=C_HIGH, lw=1.6, ls='--')
ax.set(xlabel=r'$V_\phi$ [km s$^{-1}$]', ylabel='normalised',
       title='(f) Splash cuts at $z=0$: $V_\\phi<100$ (low-$\\alpha$), $<50$ (high-$\\alpha$)')
ax.legend(fontsize=8.5)

fig.suptitle('Clumpy+merger (GASTRO c.r.c03): reproduction of Borbolato et al. (2026) Figure 5, '
             f'right-hand half  [in-situ, $R>{RMIN:.1f}$ kpc $=5\\,R_d/R_{{d,\\rm MW}}$]', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, .955])
out = G.FIG_DIR + '/gastro_fig5_clumpy_merger.png'
fig.savefig(out, dpi=140)

print('\nmedian t_form / R_form:')
for lab, m in [('low-alpha Splash ', sl), ('high-alpha Splash', sh)]:
    print(f'  {lab}  t_form={np.median(tform[m]):5.2f} Gyr  R_form={np.median(Rform[m]):5.2f} kpc  '
          f'[Fe/H]={np.median(feh[m]):+.2f}  born in the clumpy phase: '
          f'{100*np.mean(tform[m] < CLUMPY_PHASE[1]):.1f}%')
print('saved', out)
