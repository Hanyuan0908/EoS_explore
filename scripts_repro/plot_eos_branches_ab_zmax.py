"""Top two panels of obs_eos_branches_overview (halo [Mg/Fe]-[Fe/H] density, and the
same plane coloured by mean J_R), with the additional cut z_max > 2 kpc.

  (a) halo (e>0.7 | Lz<0) & z_max>2, [Mg/Fe]-[Fe/H] log-density + accreted (dashed),
      high/low-a (dotted), Eos-divider (green) lines.
  (b) same plane coloured by mean J_R (RdYlBu_r, 300-1000 kpc km/s).

J_R via AGAMA ActionFinder / McMillan17 from the AstroNN VAC (Mac-only), z_max from
the in-repo lite cache. Output figures_repro/01_eos_branches_ab_zmax2.png.
"""
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from astropy.io import fits
import agama

REPO = '/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore'
sys.path.insert(0, REPO + '/eos-figures')
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 15, 'axes.labelsize': 18, 'axes.titlesize': 17,
    'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 13,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 130, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})
FIG = REPO + '/figures_repro'
c = Cuts()

agama.setUnits(mass=1, length=1, velocity=1)
pot = agama.Potential(os.path.join(os.path.dirname(agama.__file__), 'data', 'McMillan17.ini'))
af = agama.ActionFinder(pot)

cat = load_catalog(REPO + '/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c); base = np.asarray(m['base'], bool)
feh = np.asarray(cat['fe_h'], float); mg = np.asarray(cat['mg_fe'], float)
lz = np.asarray(cat['lz'], float); rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float)
zmax = np.asarray(cat['zmax'], float)
with np.errstate(invalid='ignore'):
    ecc = (rap - rperi) / (rap + rperi)
aid = np.asarray(cat['apogee_id'])
halo = base & ((ecc > 0.7) | (lz < 0)) & np.isfinite(zmax) & (zmax > 2.0)      # + z_max>2

def acc(f):     return c.slope_acc * f + c.inter_acc
def hl(f):      return c.slope_acc2 * f + c.inter_acc2
def divline(f): return 0.317 * f + 0.353

# 6D -> J_R for the halo
ann = fits.open('/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE/apogee_astroNN-DR17.fits')[1].data
def norm(a): return np.array([(s.decode() if isinstance(s, bytes) else str(s)).strip() for s in np.asarray(a)])
nid = norm(ann['APOGEE_ID']); o = np.argsort(nid); nid_s = nid[o]
pp = np.clip(np.searchsorted(nid_s, aid), 0, len(nid_s) - 1); ok = nid_s[pp] == aid; src = o[pp]
cc = lambda n: np.where(ok, np.asarray(ann[n], float)[src], np.nan)
R, phi, zz = cc('galr'), cc('galphi'), cc('galz'); vR, vt, vz = cc('galvr'), cc('galvt'), cc('galvz')
x = R*np.cos(phi); y = R*np.sin(phi); vx = vR*np.cos(phi)-vt*np.sin(phi); vy = vR*np.sin(phi)+vt*np.cos(phi)
fin = ok & np.isfinite(x) & np.isfinite(vx) & np.isfinite(vz)
sel = halo & fin
Jr = np.full(len(feh), np.nan)
Jr[sel] = af(np.column_stack([x[sel], y[sel], zz[sel], vx[sel], vy[sel], vz[sel]]))[:, 0]

XR = (-2.1, 0.6); YMG = (-0.1, 0.5)
def mg_lines(ax):
    xx = np.linspace(*XR, 50)
    ax.plot(xx, acc(xx), color='k', ls='--', lw=1.7, zorder=3)
    ax.plot(xx, hl(xx),  color='k', ls=':',  lw=2.1, zorder=3)
    xe = np.linspace(-0.9, -0.2, 30)
    ax.plot(xe, divline(xe), color='lime', ls='-', lw=2.6, zorder=4)
def tag(a, t):
    a.text(0.03, 0.965, t, transform=a.transAxes, fontsize=18, fontweight='bold',
           va='top', ha='left', bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5))

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15.5, 6.4), constrained_layout=True)

# (a) density
s = halo & np.isfinite(feh) & np.isfinite(mg)
h, xe, ye = np.histogram2d(feh[s], mg[s], bins=[70, 55], range=[XR, YMG])
him = np.full_like(h, np.nan); him[h > 0] = np.log10(h[h > 0])
im_a = ax_a.imshow(him.T, origin='lower', extent=[*XR, *YMG], aspect='auto', cmap='Greys',
                   vmin=np.nanpercentile(him, 2), vmax=np.nanpercentile(him, 99), zorder=0)
im_a.set_rasterized(True); mg_lines(ax_a)
ax_a.text(-0.62, 0.155, 'Eos', color='k', fontsize=21, fontweight='bold', zorder=5)
ax_a.text(0.03, 0.04, r'kinematic cuts: $e>0.7$ or $L_z<0$;  $z_{\rm max}>2$ kpc',
          transform=ax_a.transAxes, fontsize=15, va='bottom', ha='left')
ax_a.set_xlim(*XR); ax_a.set_ylim(*YMG); ax_a.set_xlabel('[Fe/H]'); ax_a.set_ylabel('[Mg/Fe]')
tag(ax_a, '(a)')

# (b) mean J_R
VMIN, VMAX, NMIN = 300, 1000, 3
sJ = sel & np.isfinite(feh) & np.isfinite(mg)
med = binned_statistic_2d(feh[sJ], mg[sJ], Jr[sJ], statistic='mean', bins=(45, 35), range=[XR, YMG]).statistic
cnt = binned_statistic_2d(feh[sJ], mg[sJ], None, statistic='count', bins=(45, 35), range=[XR, YMG]).statistic
im_b = ax_b.imshow(np.where(cnt >= NMIN, med, np.nan).T, origin='lower', extent=[*XR, *YMG],
                   aspect='auto', cmap='RdYlBu_r', vmin=VMIN, vmax=VMAX, zorder=0)
im_b.set_rasterized(True); mg_lines(ax_b)
ax_b.text(-0.62, 0.155, 'Eos', color='k', fontsize=21, fontweight='bold', zorder=5)
ax_b.set_xlim(*XR); ax_b.set_ylim(*YMG); ax_b.set_xlabel('[Fe/H]'); ax_b.set_ylabel('[Mg/Fe]')
cb = fig.colorbar(im_b, ax=ax_b, pad=0.02, fraction=0.055)
cb.set_label(r'mean $J_R$ [kpc km/s]', fontsize=14); cb.ax.tick_params(labelsize=12)
tag(ax_b, '(b)')

fig.savefig(FIG + '/01_eos_branches_ab_zmax2.png', dpi=150, bbox_inches='tight')
print('wrote', FIG + '/01_eos_branches_ab_zmax2.png', ' halo&zmax>2 n=', int(halo.sum()),
      ' with J_R n=', int(np.isfinite(Jr).sum()))
