"""The v_R-v_phi story of the Eos selection, in three panels.

Uses the ORIGINAL merger-window sample -- merger_birth_vs_z0_kinematics.npz,
t_form = 4.99-6.54 Gyr, the same parent sample as ana_merger_vr_vphi_maps.py --
not the retimed window explored in ana_merger_vr_vphi_window.py.

  (a) all merger-born stars at z = 0, with the Eos band marked
  (b) the stars that pass the Eos cut, |v_phi| < 80 km/s and ecc > 0.6, at z = 0
  (c) those same stars at birth, with the v_phi,birth = 150 km/s split that
      separates born-hot (halo-born) from born-cold (disc-born)

Reading left to right: the Eos cut carves a slice out of a single rotating
distribution at z = 0, but at birth those stars were two distinct groups -- one
already slow, one on the disc ridge at ~220 km/s.  The split at 150 km/s sits in
the dip between them.

Each panel is normalised to its own peak, so colour compares shape rather than
abundance; N is annotated.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import config_au18 as C

os.makedirs(C.FIG_DIR, exist_ok=True)
VPHI_MAX, ECC_MIN, VPHI_SPLIT = 80., 0.6, 150.
RNG = [[-350, 350], [-300, 400]]
NBIN = 120
cB, cS = '#2166ac', '#111111'

k = np.load(C.OUT_DIR + '/merger_birth_vs_z0_kinematics.npz')
cat = np.load(C.OUT_DIR + '/z0_insitu_catalog.npz')
order = np.argsort(cat['ids']); sid = cat['ids'][order]
p = np.searchsorted(sid, k['ids'])
ok = (p < len(sid)) & (sid[np.minimum(p, len(sid) - 1)] == k['ids'])
ix = order[p[ok]]
bvR, bvphi = k['birth_vR'][ok], k['birth_vphi'][ok]
zvR, zvphi = k['z0_vR'][ok], k['z0_vphi'][ok]
eos = (np.abs(zvphi) < VPHI_MAX) & (cat['ecc'][ix] > ECC_MIN)
hot = eos & (bvphi < VPHI_SPLIT)
cold = eos & (bvphi >= VPHI_SPLIT)
print(f'merger-born {eos.size:,};  Eos-like {eos.sum():,};  '
      f'born hot {hot.sum():,};  born cold {cold.sum():,}')

fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.9), sharex=True, sharey=True)
panels = [(zvR, zvphi, np.ones(len(zvR), bool), 'All merger-born, $z=0$', True, False),
          (zvR, zvphi, eos, 'Eos selection, $z=0$', True, False),
          (bvR, bvphi, eos, 'The same stars at birth', False, True)]
for ax, (x, y, m, title, band, split) in zip(axes, panels):
    h, xe, ye = np.histogram2d(x[m], y[m], bins=NBIN, range=RNG)
    h = np.where(h > 0, h / h.max(), np.nan)
    im = ax.pcolormesh(xe, ye, h.T, cmap='magma_r', norm=LogNorm(vmin=1e-3, vmax=1),
                       rasterized=True)
    if band:
        ax.axhspan(-VPHI_MAX, VPHI_MAX, color=cB, alpha=.08, lw=0)
        for v in (-VPHI_MAX, VPHI_MAX):
            ax.axhline(v, color=cB, lw=1.4, ls='--')
    if split:
        ax.axhline(VPHI_SPLIT, color=cS, lw=2.2, ls='--')
        # white backing: these sit on top of the densest part of the map
        bb = dict(fc='white', ec='none', alpha=.82, pad=2.0)
        ax.annotate(f'born cold: {cold.sum():,}', (-335, VPHI_SPLIT + 26), fontsize=12,
                    va='bottom', color=cS, bbox=bb)
        ax.annotate(f'born hot: {hot.sum():,}', (-335, VPHI_SPLIT - 26), fontsize=12,
                    va='top', color=cS, bbox=bb)
        ax.annotate(r'$v_{\phi,\rm birth}=150$', (340, VPHI_SPLIT + 12), fontsize=11,
                    ha='right', va='bottom', color=cS, bbox=bb)
    ax.axhline(0, color='.55', lw=.6); ax.axvline(0, color='.55', lw=.6)
    ax.set_title(title, fontsize=13)
    ax.text(.03, .04, f'N = {m.sum():,}\n' + r'$\langle v_\phi\rangle$ = '
            + f'{np.mean(y[m]):.0f}\n' + r'$\sigma_R$ = ' + f'{np.std(x[m]):.0f}',
            transform=ax.transAxes, va='bottom', fontsize=11,
            bbox=dict(fc='white', alpha=.75, ec='none'))
    ax.set_xlabel(r'$v_R$ [km s$^{-1}$]')
axes[0].set_ylabel(r'$v_\phi$ [km s$^{-1}$]')
cb = fig.colorbar(im, ax=axes, fraction=.021, pad=.012)
cb.set_label("density, normalised to each panel's own peak")
fig.suptitle('Au18: how the Eos selection looks today, and what those stars were at birth '
             f'($t_{{\\rm form}}$ = 4.99–6.54 Gyr)', fontsize=13.5)
out = C.FIG_DIR + '/au18_merger_vr_vphi_three.png'
fig.savefig(out, dpi=145, bbox_inches='tight')
print('saved', out)
