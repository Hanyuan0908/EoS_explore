"""Au18: the v_R - v_phi plane of stars formed during the GS/E merger.

Rebuild of ana_merger_birth_vs_z0_kinematics.py, which showed the merger-born
sample as a whole and was therefore dominated by the rotating disc: <v_phi> only
falls from 209 to 201 km/s between birth and z=0, and the Eos-like population is
invisible under the disc.  Here the same stars are shown with the pieces we have
since derived overlaid, so the interesting population can actually be seen:

  * the observational Eos box, -80 < v_phi < +80 km/s -- this project's own
    symmetric Splash/Eos window (SPLASH_VTAN_MAX in ../src/eos/config.py), the
    same one used for the symmetric variant of the Fig. 5 reproduction, rather
    than the looser |v_phi| < 100 used in ana_eos_age_kinematics.py;
  * the GS/E debris, which is the accreted comparison;
  * contours of the merger-born stars that END UP inside the Eos box, drawn on
    the BIRTH plane -- this is the born-hot versus heated question in one panel.

Everything is read from cached arrays; no snapshots are touched.

Reads out/merger_birth_vs_z0_kinematics.npz and out/z0_insitu_catalog.npz.
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

os.makedirs(C.FIG_DIR, exist_ok=True)
VPHI_MAX, ECC_MIN = 80., 0.6      # symmetric window: -80 < v_phi < +80 km/s
RNG = [[-350, 350], [-300, 400]]
C_EOS, C_GSE = '#b2182b', '#1b7837'

k = np.load(C.OUT_DIR + '/merger_birth_vs_z0_kinematics.npz')
cat = np.load(C.OUT_DIR + '/z0_insitu_catalog.npz')

# Match the merger-born stars into the z=0 catalogue for ecc / [Fe/H] / t_form.
order = np.argsort(cat['ids']); sid = cat['ids'][order]
p = np.searchsorted(sid, k['ids'])
ok = (p < len(sid)) & (sid[np.minimum(p, len(sid) - 1)] == k['ids'])
ix = order[p[ok]]
bvR, bvphi = k['birth_vR'][ok], k['birth_vphi'][ok]
zvR, zvphi = k['z0_vR'][ok], k['z0_vphi'][ok]
ecc, tform, feh = cat['ecc'][ix], cat['tform'][ix], cat['feh'][ix]

# "Eos-like at z=0" on the same footing as the rest of the project.
eos = (np.abs(zvphi) < VPHI_MAX) & (ecc > ECC_MIN)
print(f'merger-born stars matched at z=0: {len(bvR):,} '
      f'(t_form {tform.min():.2f}-{tform.max():.2f} Gyr)')
print(f'  of these, Eos-like at z=0 ({-VPHI_MAX:.0f}<v_phi<{VPHI_MAX:.0f}, ecc>{ECC_MIN}): '
      f'{eos.sum():,} ({100*eos.mean():.2f}%)')
print(f'  their v_phi at birth: median {np.median(bvphi[eos]):.1f} km/s '
      f'(whole sample {np.median(bvphi):.1f})')

g_ok = np.isfinite(cat['gse_vphi'])
fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.9))


for ax, (x, y, title) in zip(axes[:2], [
        (bvR, bvphi, 'Near birth (during the merger)'),
        (zvR, zvphi, 'Present day ($z=0$)')]):
    ax.hist2d(x, y, bins=140, range=RNG, cmap='Greys', cmin=1, norm=LogNorm())
    ax.axhspan(-VPHI_MAX, VPHI_MAX, color=C_EOS, alpha=.07, lw=0)
    for v in (-VPHI_MAX, VPHI_MAX):
        ax.axhline(v, color=C_EOS, lw=1.2, ls='--')
    ax.axhline(0, color='.6', lw=.6)
    ax.axvline(0, color='.6', lw=.6)
    # the subset that ends up Eos-like, on both planes
    OT.density_contours(ax, x[eos], y[eos], RNG, C_EOS, label='ends up Eos-like',
                        levels=(0.9, 0.6, 0.3), bins=70)
    ax.set(xlabel=r'$v_R$ [km s$^{-1}$]', ylabel=r'$v_\phi$ [km s$^{-1}$]',
           xlim=RNG[0], ylim=RNG[1], title=title)
    ax.text(.03, .04, f'N={len(x):,}\n' + r'$\langle v_\phi\rangle$=' + f'{np.mean(y):.0f}\n'
            + r'$\sigma_R$=' + f'{np.std(x):.0f}', transform=ax.transAxes, fontsize=9,
            bbox=dict(fc='white', alpha=.85, ec='none'))
    ax.legend(fontsize=9, loc='upper left')

# third panel: z=0 with the accreted debris for scale
ax = axes[2]
ax.hist2d(zvR, zvphi, bins=140, range=RNG, cmap='Greys', cmin=1, norm=LogNorm())
OT.density_contours(ax, zvR[eos], zvphi[eos], RNG, C_EOS, label='merger-born, Eos-like',
                    levels=(0.9, 0.6, 0.3), bins=70)
OT.density_contours(ax, cat['gse_vR'][g_ok], cat['gse_vphi'][g_ok], RNG, C_GSE,
                    label='GS/E debris (accreted)', levels=(0.9, 0.6, 0.3), bins=70, ls='--')
for v in (-VPHI_MAX, VPHI_MAX):
    ax.axhline(v, color=C_EOS, lw=1.2, ls='--')
ax.axhline(0, color='.6', lw=.6); ax.axvline(0, color='.6', lw=.6)
ax.set(xlabel=r'$v_R$ [km s$^{-1}$]', ylabel=r'$v_\phi$ [km s$^{-1}$]',
       xlim=RNG[0], ylim=RNG[1], title='$z=0$, against the accreted GS/E debris')
ax.legend(fontsize=9, loc='upper left')

fig.suptitle('Au18: $v_R$-$v_\\phi$ of the in-situ stars formed during the GS/E merger '
             f'($t_{{\\rm form}}={tform.min():.1f}$-{tform.max():.1f} Gyr); '
             f'shaded band = the ${-VPHI_MAX:.0f}<v_\\phi<{VPHI_MAX:.0f}$ km/s Eos cut', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, .93])
out = C.FIG_DIR + '/au18_merger_vr_vphi.png'
fig.savefig(out, dpi=145)

print(f'\n{"":22s} {"<v_phi>":>9s} {"sigma_R":>9s} {"sigma_phi":>10s}')
for lab, x, y in [('all merger-born, birth', bvR, bvphi),
                  ('all merger-born, z=0  ', zvR, zvphi),
                  ('-> Eos-like, at birth ', bvR[eos], bvphi[eos]),
                  ('-> Eos-like, at z=0   ', zvR[eos], zvphi[eos]),
                  ('GS/E debris, z=0      ', cat['gse_vR'][g_ok], cat['gse_vphi'][g_ok])]:
    print(f'{lab:22s} {np.mean(y):9.1f} {np.std(x):9.1f} {np.std(y):10.1f}')
print('saved', out)
