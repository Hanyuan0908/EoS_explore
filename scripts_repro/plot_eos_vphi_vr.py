"""Two-panel velocity plane of the two Eos branches, axes SWAPPED (V_R on x, V_phi on y):
  LEFT  -- coloured by branch (alpha-rich upper / alpha-poor lower), grey low-alpha bg.
  RIGHT -- same points coloured by [Fe/H].
Eos = thin_al & -0.9<[Fe/H]<-0.5, halo via ecc>0.7 (no vphi bias); divider mg=0.317*feh+0.353.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
REPO = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/eos-figures')
sys.path.insert(0, str(REPO))
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
from eos_figures.plotting import label_axes
c = Cuts()
FIG = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/figures_repro')
cat = load_catalog('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c)
feh = np.asarray(cat['fe_h'], float); mg = np.asarray(cat['mg_fe'], float)
al = np.asarray(cat['al_fe'], float); lz = np.asarray(cat['lz'], float)
vphi = np.asarray(cat['galvt'], float)
rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float)
ecc = (rap - rperi) / (rap + rperi); aid = np.asarray(cat['apogee_id'])
base = np.asarray(m['base'], bool); thin_al = np.asarray(m['thin_al'], bool)

ann = fits.open('/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE/apogee_astroNN-DR17.fits')[1].data
def norm(a): return np.array([(s.decode() if isinstance(s, bytes) else str(s)).strip() for s in np.asarray(a)])
nid = norm(ann['APOGEE_ID']); o = np.argsort(nid); nid_s = nid[o]
p = np.clip(np.searchsorted(nid_s, aid), 0, len(nid_s)-1); ok = nid_s[p] == aid; src = o[p]
vR = np.where(ok, np.asarray(ann['galvr'], float)[src], np.nan)
fin = ok & np.isfinite(vR) & np.isfinite(vphi)

# CANONICAL Eos cut (matches the bifurcation figure): Davies halo & low-alpha wedge & -0.9<[Fe/H]<-0.2
halo = base & ((ecc > 0.7) | (lz < 0))
def acc(f): return c.slope_acc*f + c.inter_acc
def hl(f): return c.slope_acc2*f + c.inter_acc2
def divline(f): return 0.317 * f + 0.353
lowa = halo & (feh > -0.9) & (feh < -0.2) & (mg > acc(feh)) & (mg < hl(feh)) & (al > c.alfe_cut) & fin
eos = lowa
eos_hi = eos & (mg > divline(feh))
eos_lo = eos & (mg <= divline(feh))
CU, CL = '#e07a1f', '#2b6cb0'
XR, YR = (-250, 250), (-250, 400)          # x = V_R, y = V_phi

fig, ax = plt.subplots(1, 2, figsize=(14.5, 7.2), constrained_layout=True)

# LEFT: coloured by branch
bg = thin_al & fin
ax[0].hist2d(vR[bg], vphi[bg], bins=[120, 120], range=[XR, YR], cmap='Greys', norm=LogNorm(), zorder=0)
ax[0].scatter(vR[eos_hi], vphi[eos_hi], s=26, c=CU, edgecolors='k', linewidths=0.3,
              label=fr'Eos $\alpha$-rich (upper), n={int(eos_hi.sum())}', zorder=3)
ax[0].scatter(vR[eos_lo], vphi[eos_lo], s=26, c=CL, edgecolors='k', linewidths=0.3,
              label=fr'Eos $\alpha$-poor (lower), n={int(eos_lo.sum())}', zorder=3)
for sel, col in [(eos_hi, CU), (eos_lo, CL)]:
    ax[0].scatter(vR[sel].mean(), vphi[sel].mean(), s=320, marker='*', c=col,
                  edgecolors='k', linewidths=1.3, zorder=5)
ax[0].axhline(0, color='0.4', lw=0.8, ls='--'); ax[0].axvline(0, color='0.4', lw=0.8, ls='--')
ax[0].set_xlim(*XR); ax[0].set_ylim(*YR)
label_axes(ax[0], r'$V_R$ [km/s]', r'$V_\phi\ (\equiv V_{\rm tan})$ [km/s]', 'coloured by branch')
ax[0].legend(frameon=False, fontsize=9.5, loc='upper left')

# RIGHT: background density coloured by MEAN [Fe/H] per bin (not the scatter)
NB = 90; NMIN = 5
cnt, xe, ye = np.histogram2d(vR[bg], vphi[bg], bins=NB, range=[XR, YR])
fsum, _, _ = np.histogram2d(vR[bg], vphi[bg], bins=NB, range=[XR, YR], weights=feh[bg])
mfeh = np.full_like(cnt, np.nan); ok2 = cnt >= NMIN; mfeh[ok2] = fsum[ok2] / cnt[ok2]
im = ax[1].imshow(mfeh.T, origin='lower', extent=[*XR, *YR], aspect='auto', cmap='coolwarm',
                  vmin=-0.7, vmax=0.1, zorder=0)
ax[1].scatter(vR[eos_hi], vphi[eos_hi], s=22, c=CU, edgecolors='k', linewidths=0.3,
              label=fr'Eos $\alpha$-rich', zorder=3)
ax[1].scatter(vR[eos_lo], vphi[eos_lo], s=22, c=CL, edgecolors='k', linewidths=0.3,
              label=fr'Eos $\alpha$-poor', zorder=3)
ax[1].axhline(0, color='0.4', lw=0.8, ls='--'); ax[1].axvline(0, color='0.4', lw=0.8, ls='--')
ax[1].set_xlim(*XR); ax[1].set_ylim(*YR)
fig.colorbar(im, ax=ax[1], pad=0.01).set_label(r'mean [Fe/H] of low-$\alpha$ stars in bin [dex]')
ax[1].legend(frameon=False, fontsize=9.5, loc='upper left')
label_axes(ax[1], r'$V_R$ [km/s]', r'$V_\phi\ (\equiv V_{\rm tan})$ [km/s]', r'background density coloured by [Fe/H]')

fig.suptitle(r'Eos velocity plane ($V_\phi$ vs $V_R$): two chemical branches, one dispersion-dominated kinematics', fontsize=13)
fig.savefig(FIG / '01_eos_vphi_vr.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_eos_vphi_vr.png')
