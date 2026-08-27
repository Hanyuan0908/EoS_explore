"""R_birth against [Fe/H] as contours, for the two Eos populations.

The earlier version (ana_eos_origins_rbirth.py) showed only running medians,
which hide how much the two populations overlap and how wide each one is.  Here
they are drawn as density contours instead.

R_birth is the instantaneous cylindrical radius at the nearest snapshot after
formation, NOT the guiding radius: the two answer different questions.  R_g says
which orbit a star belongs to, R_birth says where it physically was when it
formed, and for chemistry -- fixed at birth from the gas in that place -- R_birth
is the relevant one.

Left panel takes the whole formation window; right restricts to the merger
proper, t_form = 5.0-5.45 Gyr, from the pericentre plunge to coalescence, which
holds 64% of the halo-born stars and 48% of the disc-born.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import orbit_tools as OT
import config_au18 as C
import eos_origins as EO

os.makedirs(C.FIG_DIR, exist_ok=True)
d = EO.load()
C_HALO, C_DISC = '#7b3294', 'crimson'
RNG = [[0, 22], [-1.6, 0.5]]
T_LO, T_HI = 5.0, 5.45

tf = d['tform']
finite = np.isfinite(d['R_birth']) & np.isfinite(d['feh'])
WINDOWS = [(np.ones(len(tf), bool), f'Whole formation window ($t_{{\\rm form}}$ = 4.99-6.54 Gyr)'),
           ((tf >= T_LO) & (tf <= T_HI), f'The merger itself ($t_{{\\rm form}}$ = {T_LO}-{T_HI} Gyr, '
                                         'plunge to coalescence)')]
REDGE = np.concatenate([np.arange(0, 16, 1.5), [18, 22]])
RCEN = .5 * (REDGE[:-1] + REDGE[1:])


def running_median(x, y, mask, nmin=15):
    out = np.full(len(RCEN), np.nan)
    good = mask & np.isfinite(x) & np.isfinite(y)
    for i in range(len(RCEN)):
        s = good & (x >= REDGE[i]) & (x < REDGE[i + 1])
        if s.sum() >= nmin:
            out[i] = np.median(y[s])
    return out


fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.4))
for ax, (win, title) in zip(axes, WINDOWS):
    bg = finite & win
    ax.hist2d(d['R_birth'][bg], d['feh'][bg], bins=(120, 100), range=RNG,
              norm=LogNorm(), cmap='Greys')
    for lab, m, c in [('disc-born Eos', d['disc_born'], C_DISC),
                      ('halo-born Eos (merger-triggered)', d['halo_born'], C_HALO)]:
        sel = m & win & finite
        OT.density_contours(ax, d['R_birth'][sel], d['feh'][sel], RNG, c,
                            label=f'{lab} ({sel.sum():,})', levels=(0.9, 0.6, 0.3),
                            bins=55, lw=2.1)
        ax.plot(RCEN, running_median(d['R_birth'], d['feh'], sel), 'o-', color=c,
                lw=1.3, ms=3.5, alpha=.75)
    ax.plot(RCEN, running_median(d['R_birth'], d['feh'], bg), 's--', color='k', lw=1.6, ms=4,
            label='all stars born in this window')
    ax.set(xlim=RNG[0], ylim=RNG[1], xlabel=r'$R_{\rm birth}$ [kpc]', ylabel='[Fe/H]',
           title=title)
    ax.legend(fontsize=9, loc='upper right')

fig.suptitle('Au18: birth radius against metallicity for the two Eos populations '
             '(contours enclose 30, 60 and 90 per cent; thin lines are running medians)',
             fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, .94])
out = C.FIG_DIR + '/au18_eos_origins_rbirth_contour.png'
fig.savefig(out, dpi=145)

for win, title in WINDOWS:
    print(f'\n{title.replace("$", "").replace(chr(92) + "rm ", "")}')
    print(f'{"":34s} {"N":>7s} {"R_birth p16/50/84":>26s} {"[Fe/H] p16/50/84":>26s}')
    for lab, m in [('disc-born Eos', d['disc_born']), ('halo-born Eos', d['halo_born']),
                   ('all born in this window', np.ones(len(tf), bool))]:
        s = m & win & finite
        rq = np.percentile(d['R_birth'][s], [16, 50, 84])
        fq = np.percentile(d['feh'][s], [16, 50, 84])
        print(f'  {lab:32s} {s.sum():7,} '
              f'{rq[0]:7.2f} {rq[1]:7.2f} {rq[2]:7.2f}   {fq[0]:+7.2f} {fq[1]:+7.2f} {fq[2]:+7.2f}')
    a = running_median(d['R_birth'], d['feh'], d['halo_born'] & win & finite)
    b = running_median(d['R_birth'], d['feh'], d['disc_born'] & win & finite)
    both = np.isfinite(a) & np.isfinite(b)
    if both.any():
        print(f'  [Fe/H] offset at matched R_birth: {np.mean((a - b)[both]):+.3f} dex '
              f'over {both.sum()} shared bins')
print('\nsaved', out)
