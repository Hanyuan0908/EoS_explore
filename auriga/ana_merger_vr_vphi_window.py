"""Au18: side-by-side density maps of the merger-born stars and their Eos-like subset.

The companion to ana_merger_vr_vphi.py, which overlays contours on one map.  Here
the two populations get their own panels so the shapes can be compared directly,
at birth and at z=0.

Each panel is normalised to its own peak, because the samples differ by a factor
of ~23 in number (171,826 against 7,583) and a shared absolute scale would render
the Eos-like maps nearly blank.  The colour therefore shows the shape of each
distribution, not its amplitude.

"All" here means all in-situ stars formed during the merger window
New-window version: t_form is set by eos_origins_window.WINDOW (4.7-5.7 Gyr),
not the 4.99-6.54 Gyr parent sample of ana_merger_vr_vphi_maps.py.  Writes to
*_win.png so both versions coexist.

Reads out/merger_birth_vs_z0_kinematics.npz and out/z0_insitu_catalog.npz.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_au18 as C

os.makedirs(C.FIG_DIR, exist_ok=True)
VPHI_MAX, ECC_MIN = 80., 0.6
RNG = [[-350, 350], [-300, 400]]
NBIN = 120

import eos_origins_window as EOW
WIN = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else EOW.WINDOW
d = EOW.load(WIN)
bvR, bvphi, zvR, zvphi, eos = d['bvR'], d['bvphi'], d['zvR'], d['zvphi'], d['eos']
print(f'window {WIN[0]}-{WIN[1]} Gyr: N={len(bvR):,}, Eos-like {eos.sum():,}')

fig, axes = plt.subplots(2, 2, figsize=(13, 10.2), sharex=True, sharey=True)
panels = [
    (0, 0, bvR, bvphi, np.ones(len(bvR), bool), f'All born {WIN[0]}-{WIN[1]} Gyr', 'Near birth'),
    (0, 1, bvR, bvphi, eos, 'Eos-like subset', 'Near birth'),
    (1, 0, zvR, zvphi, np.ones(len(zvR), bool), f'All born {WIN[0]}-{WIN[1]} Gyr', 'Present day ($z=0$)'),
    (1, 1, zvR, zvphi, eos, 'Eos-like subset', 'Present day ($z=0$)'),
]
for r, c, x, y, m, who, when in panels:
    ax = axes[r, c]
    h, xe, ye = np.histogram2d(x[m], y[m], bins=NBIN, range=RNG)
    h = np.where(h > 0, h / h.max(), np.nan)          # each panel to its own peak
    im = ax.pcolormesh(xe, ye, h.T, cmap='magma_r', norm=LogNorm(vmin=1e-3, vmax=1))
    ax.axhspan(-VPHI_MAX, VPHI_MAX, color='#2166ac', alpha=.07, lw=0)
    for v in (-VPHI_MAX, VPHI_MAX):
        ax.axhline(v, color='#2166ac', lw=1.2, ls='--')
    ax.axhline(0, color='.5', lw=.6); ax.axvline(0, color='.5', lw=.6)
    ax.set_title(f'{who} — {when}', fontsize=12)
    ax.text(.03, .04, f'N={m.sum():,}\n' + r'$\langle v_\phi\rangle$=' + f'{np.mean(y[m]):.0f}\n'
            + r'$\sigma_R$=' + f'{np.std(x[m]):.0f}',
            transform=ax.transAxes, fontsize=9.5,
            bbox=dict(fc='white', alpha=.85, ec='none'))
    if r == 1:
        ax.set_xlabel(r'$v_R$ [km s$^{-1}$]')
    if c == 0:
        ax.set_ylabel(r'$v_\phi$ [km s$^{-1}$]')
ax.set(xlim=RNG[0], ylim=RNG[1])

cb = fig.colorbar(im, ax=axes, fraction=.028, pad=.02)
cb.set_label('density, normalised to each panel\'s own peak')
fig.suptitle('Au18: $v_R$–$v_\\phi$ of stars formed during the GS/E merger '
             f'($t_{{\\rm form}}$ = {WIN[0]}–{WIN[1]} Gyr).  '
             f'Band = the ${-VPHI_MAX:.0f}<v_\\phi<{VPHI_MAX:.0f}$ km/s Eos cut', fontsize=13)
out = C.FIG_DIR + '/au18_merger_vr_vphi_maps_win.png'
fig.savefig(out, dpi=145, bbox_inches='tight')

print(f'{"":26s} {"N":>9s} {"<v_phi>":>9s} {"sigma_R":>9s} {"sigma_phi":>10s}')
for _, _, x, y, m, who, when in panels:
    print(f'{who + ", " + when.replace("$", "").replace("z=0", "z=0"):26s} '
          f'{m.sum():9,} {np.mean(y[m]):9.1f} {np.std(x[m]):9.1f} {np.std(y[m]):10.1f}')
print('saved', out)
