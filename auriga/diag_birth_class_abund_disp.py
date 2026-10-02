"""Diagnostic: chemical dispersion of the two birth classes at the GS/E pericentre.

Same sample as au18_birth_positions_gas4 -- the stars formed in the snapshot-72
window (t_form = 4.86-4.99 Gyr), split by the birth orbit they were put on:

  disc-born   eps > 0.8  OR  z_max < 1.5 kpc
  halo-born   eps <= 0.8 AND z_max >= 1.5 kpc

For each of the six metals Auriga tracks besides Fe, the robust dispersion
sigma_[X/Fe] = 1.4826 x MAD is measured in FIXED 0.1 dex bins of [Fe/H], so the
two classes are compared at the same metallicity rather than at their own
different medians.  Bands are the 16-84 range of 400 bootstrap resamples of the
dispersion itself, which is what says whether a difference between the two
curves is real at the N of that bin.

Why fixed bins matter here: the two classes barely overlap in [Fe/H] -- median
+0.156 for disc-born against -0.458 for halo-born, 0.61 dex apart -- and
sigma_[X/Fe] in this model falls steeply above [Fe/H] ~ -0.2 for both classes.
An unbinned comparison would hand that gradient back as if it were a difference
between the classes.  Only five 0.2 dex bins, roughly -0.6 to +0.4, hold at
least NMIN of each class; outside them one curve is an extrapolation of a
population the other does not reach, which is why the summary panel counts the
comparable bins.

RESULT: at fixed [Fe/H] the halo-born class is the chemically more inhomogeneous
one in all six elements, by +0.002 (O) to +0.019 (Mg) dex N-weighted, and the gap
is widest in Mg, N and C.  The two curves converge above [Fe/H] ~ +0.2, where the
halo-born class is down to its metal-rich tail.

CAVEAT -- Auriga's abundances are not the Milky Way's.  The [Mg/Fe] zero point is
offset (the galaxy sits near -0.30, not 0.0) and the 16-84 spread of every alpha
element is 0.03-0.05 dex against the ~0.3 dex separating the MW's two sequences.
Read the dispersions as a measure of ISM inhomogeneity in the model, not as
something to compare against an observed sigma_[X/Fe].

Writes figures/au18_birth_class_abund_disp.png.
"""
import gc, os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import auriga_public as ap
import config_au18 as C

SNAP = int(sys.argv[1]) if len(sys.argv) > 1 else 72
CUT, ZCUT = 0.8, 1.5
ELS = ['C', 'N', 'O', 'Ne', 'Mg', 'Si']
# 0.2 dex bins, not 0.1: at 0.1 the curves carried bin-to-bin wiggles comparable
# to the bootstrap band, which read as structure they do not have.  Doubling the
# width roughly halves the scatter on each point and leaves the disc/halo
# separation -- the thing the figure is for -- unchanged.  Override on the
# command line: diag_birth_class_abund_disp.py [snap] [bin width]
FBIN_W = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
FBINS = np.arange(-1.2, 0.6 + 1e-9, FBIN_W)
NMIN = 50                      # a bin is drawn only if BOTH classes reach this
NBOOT = 400
RNG = np.random.default_rng(0)
cD, cH = '#1F6FB2', '#FF6347'
OUT = os.path.dirname(os.path.abspath(__file__)) + '/figures'
os.makedirs(OUT, exist_ok=True)

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 13, 'axes.labelsize': 14.5,
    'xtick.labelsize': 12.5, 'ytick.labelsize': 12.5, 'legend.fontsize': 12,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 140, 'savefig.dpi': 200,
})


def mad_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def disp_profile(feh, y, m):
    """Robust sigma(y) per [Fe/H] bin for the mask m, with bootstrap 16-84."""
    n = len(FBINS) - 1
    s = np.full(n, np.nan); lo = np.full(n, np.nan); hi = np.full(n, np.nan)
    cnt = np.zeros(n, int)
    ok = m & np.isfinite(feh) & np.isfinite(y)
    for k in range(n):
        b = ok & (feh >= FBINS[k]) & (feh < FBINS[k + 1])
        cnt[k] = b.sum()
        if cnt[k] < NMIN:
            continue
        v = y[b]
        s[k] = mad_sigma(v)
        bs = np.array([mad_sigma(v[RNG.integers(0, len(v), len(v))])
                       for _ in range(NBOOT)])
        lo[k], hi[k] = np.percentile(bs, (16, 84))
    return s, lo, hi, cnt


# ---- sample: stars formed in the snapshot-SNAP window, classified at birth ----
a = np.load(C.OUT_DIR + '/birth_orbits_actions.npz')
zx = np.load(C.OUT_DIR + '/birth_orbits_zmax.npz')
st = np.load(C.OUT_DIR + '/snapshot_times.npz')
SN_ALL, T_ALL = st['snaps'], st['t_snap']
k = int(np.flatnonzero(SN_ALL == SNAP)[0])
t_lo, t_hi = T_ALL[k - 1], T_ALL[k]

sel = (a['tform'] > t_lo) & (a['tform'] <= t_hi) & np.isfinite(a['eps_birth'])
ids_w, eps_w, zmx_w = a['ids'][sel], a['eps_birth'][sel], zx['zmax_birth'][sel]

s = ap.snapshot.load_snapshot(SNAP, 4, snappath=C.SIM_DIR, verbose=False,
                              loadlist=['ParticleIDs', 'GFM_Metals'])
sids, met = s.data['ParticleIDs'], s.data['GFM_Metals']
del s; gc.collect()

o = np.argsort(sids); ss = sids[o]
p = np.searchsorted(ss, ids_w)
ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == ids_w)
M = met[o[p[ok]]]
eps, zmx = eps_w[ok], zmx_w[ok]
disc = (eps > CUT) | (zmx < ZCUT)
del met; gc.collect()

feh = C.bracket_abundance(M, 'Fe', 'H')
XFE = {e: C.bracket_abundance(M, e, 'Fe') for e in ELS}
POPS = [('disc-born', disc, cD), ('halo-born', ~disc, cH)]

print(f'snap {SNAP}: t_form window {t_lo:.2f}-{t_hi:.2f} Gyr, matched {ok.sum():,}')
for lab, m, _ in POPS:
    print(f'  {lab:10s} N={m.sum():>7,}  median [Fe/H] = {np.nanmedian(feh[m]):+.3f}')

# ---------------------------------- figure ----------------------------------
fig, axes = plt.subplots(2, 4, figsize=(17.4, 7.8))
ax0 = axes[0, 0]
for lab, m, c in POPS:
    ax0.hist(feh[m], bins=FBINS, density=True, histtype='step', color=c, lw=2.4)
    ax0.axvline(np.nanmedian(feh[m]), color=c, lw=1.2, ls='--')
ax0.set(xlabel='[Fe/H]', ylabel='density', xlim=(FBINS[0], FBINS[-1]))
ax0.text(.035, .95, '(a)  where each class lives', transform=ax0.transAxes,
         va='top', fontsize=12.5)

ctr = .5 * (FBINS[:-1] + FBINS[1:])
tags = 'bcdefg'
rows = []
for i, el in enumerate(ELS):
    ax = axes.flat[i + 1]
    prof = {}
    for lab, m, c in POPS:
        sg, lo, hi, cnt = disp_profile(feh, XFE[el], m)
        prof[lab] = (sg, lo, hi, cnt)
        good = np.isfinite(sg)
        ax.plot(ctr[good], sg[good], color=c, lw=2.4, label=lab)
        ax.fill_between(ctr[good], lo[good], hi[good], color=c, alpha=.22, lw=0)
    # a bin is only comparable where both classes cleared NMIN
    both = np.isfinite(prof['disc-born'][0]) & np.isfinite(prof['halo-born'][0])
    ax.set(xlabel='[Fe/H]', ylabel=rf'$\sigma_{{[\rm {el}/Fe]}}$ [dex]',
           xlim=(FBINS[0], FBINS[-1]))
    ax.text(.035, .95, f'({tags[i]})  {el}', transform=ax.transAxes, va='top',
            fontsize=13, fontweight='bold')
    if both.any():
        d = prof['halo-born'][0][both] - prof['disc-born'][0][both]
        w = prof['disc-born'][3][both] + prof['halo-born'][3][both]
        rows.append((el, np.average(d, weights=w), both.sum()))
        print(f'  {el:2s}: {both.sum()} comparable bins, '
              f'N-weighted halo-disc = {np.average(d, weights=w):+.4f} dex')

axes[0, 1].legend(loc='upper right', handlelength=1.6)

axl = axes[1, 3]
axl.axis('off')
txt = ('N-weighted $\\sigma_{\\rm halo} - \\sigma_{\\rm disc}$\n'
       'over the bins both classes fill:\n\n'
       + '\n'.join(f'   {el:<3s} {d:+.4f} dex   ({n} bins)' for el, d, n in rows)
       + f'\n\nfixed bins of {FBIN_W:g} dex, $N \\geq {NMIN}$ per class\n'
         f'bands: 16-84 of {NBOOT} bootstraps\n'
         f'$t_{{\\rm form}} = {t_lo:.2f}-{t_hi:.2f}$ Gyr (snapshot {SNAP})')
axl.text(.0, .97, txt, transform=axl.transAxes, va='top', ha='left', fontsize=12,
         family='monospace', linespacing=1.45)

fig.tight_layout(pad=.6, w_pad=1.1, h_pad=1.0)
f = f'{OUT}/au18_birth_class_abund_disp.png'
fig.savefig(f, bbox_inches='tight')
print('\nsaved', f)
