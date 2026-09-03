"""Same Davies halo chemistry planes, coloured by ORBITAL CIRCULARITY eta = Lz / L_circ(E),
the properly-normalized rotation measure: +1 prograde-circular (disc), 0 non-rotating
(radial/polar halo), -1 retrograde-circular. Bounded [-1,1] -> diverging coolwarm about 0.
L_circ(E) from AGAMA (circular-orbit radius at each star's energy) in the McMillan 2017 potential.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from astropy.io import fits
import agama
REPO = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/eos-figures')
sys.path.insert(0, str(REPO))
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
from eos_figures.plotting import label_axes
c = Cuts()
FIG = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/figures_repro')
agama.setUnits(mass=1, length=1, velocity=1)
pot = agama.Potential(os.path.join(os.path.dirname(agama.__file__), 'data', 'McMillan17.ini'))

cat = load_catalog('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c); base = np.asarray(m['base'], bool)
feh = np.asarray(cat['fe_h'], float); mg = np.asarray(cat['mg_fe'], float); al = np.asarray(cat['al_fe'], float)
lz = np.asarray(cat['lz'], float); rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float)
ecc = (rap - rperi)/(rap + rperi); aid = np.asarray(cat['apogee_id'])
halo = base & ((ecc > 0.7) | (lz < 0))

ann = fits.open('/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE/apogee_astroNN-DR17.fits')[1].data
def norm(a): return np.array([(s.decode() if isinstance(s, bytes) else str(s)).strip() for s in np.asarray(a)])
nid = norm(ann['APOGEE_ID']); o = np.argsort(nid); nid_s = nid[o]
p = np.clip(np.searchsorted(nid_s, aid), 0, len(nid_s)-1); ok = nid_s[p] == aid; src = o[p]
cc = lambda n: np.where(ok, np.asarray(ann[n], float)[src], np.nan)
R, phi, zz = cc('galr'), cc('galphi'), cc('galz'); vR, vt, vz = cc('galvr'), cc('galvt'), cc('galvz')
x = R*np.cos(phi); y = R*np.sin(phi); vx = vR*np.cos(phi)-vt*np.sin(phi); vy = vR*np.sin(phi)+vt*np.cos(phi)
fin = ok & np.isfinite(x) & np.isfinite(vx) & np.isfinite(vz)
sel = halo & fin

# circularity eta = Lz / L_circ(E)
xyz = np.column_stack([x[sel], y[sel], zz[sel]])
E = 0.5*(vx[sel]**2 + vy[sel]**2 + vz[sel]**2) + pot.potential(xyz)
Rc = pot.Rcirc(E=E)                                          # radius of circular orbit with this energy
fR = pot.force(np.column_stack([Rc, 0*Rc, 0*Rc]))[:, 0]     # radial force at Rc
Lcirc = Rc * np.sqrt(-Rc * fR)                              # = Rc * vcirc(Rc)
Lz_star = x[sel]*vy[sel] - y[sel]*vx[sel]
eta = np.full(len(feh), np.nan); eta[sel] = Lz_star / Lcirc
print('circularity eta percentiles 5/50/95:', np.round(np.nanpercentile(eta[sel], [5, 50, 95]), 2),
      ' frac retrograde (eta<0):', round(np.mean(eta[sel] < 0), 2))

STAT = sys.argv[1] if len(sys.argv) > 1 else 'mean'
XR = (-2.1, 0.6); NMIN = 3
VMIN, VMAX = -0.35, 0.35                            # eta colour range (diverging about 0)
def panel(ax, yv, yr, ylab, nb=(45, 35)):
    s = sel & np.isfinite(feh) & np.isfinite(yv)
    me = binned_statistic_2d(feh[s], yv[s], eta[s], statistic=STAT, bins=nb, range=[XR, yr]).statistic
    cnt = binned_statistic_2d(feh[s], yv[s], None, statistic='count', bins=nb, range=[XR, yr]).statistic
    mj = np.where(cnt >= NMIN, me, np.nan)
    im = ax.imshow(mj.T, origin='lower', extent=[*XR, *yr], aspect='auto', cmap='coolwarm',
                   vmin=VMIN, vmax=VMAX, zorder=0)
    ax.set_xlim(*XR); ax.set_ylim(*yr)
    label_axes(ax, '[Fe/H]', ylab, f'Davies halo, coloured by {STAT} circularity $\\eta=L_z/L_c$ (n={int(s.sum())})')
    return im

def mg_lines(ax):
    xx = np.linspace(-2.1, 0.6, 50)
    ax.plot(xx, c.slope_acc*xx + c.inter_acc, color='k', ls='--', lw=1.6, zorder=3, label='accreted / in-situ')
    ax.plot(xx, c.slope_acc2*xx + c.inter_acc2, color='k', ls=':', lw=2.0, zorder=3, label=r'high-$\alpha$ / low-$\alpha$')
    xe = np.linspace(-0.9, -0.2, 30)
    ax.plot(xe, 0.317*xe + 0.353, color='lime', ls='-', lw=2.0, zorder=4, label=r'Eos divider ($\alpha$-rich/$\alpha$-poor)')
    leg = ax.legend(frameon=False, fontsize=8, loc='lower left')
    for t in leg.get_texts(): t.set_color('black')

fig, ax = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
panel(ax[0], al, (-0.45, 0.45), '[Al/Fe]')
im = panel(ax[1], mg, (-0.1, 0.5), '[Mg/Fe]')
mg_lines(ax[1])
ax[1].text(-0.62, 0.16, 'Eos', color='k', fontsize=13, fontweight='bold', zorder=3)
cb = fig.colorbar(im, ax=ax, pad=0.01, shrink=0.9); cb.set_label(f'{STAT} circularity ' + r'$\eta = L_z/L_{\rm circ}(E)$')
fig.suptitle(f'Orbital circularity ({STAT}) across the halo chemistry planes (AGAMA / McMillan 2017)', fontsize=12)
out = '01_davies_fig2_circ.png' if STAT == 'mean' else f'01_davies_fig2_circ_{STAT}.png'
fig.savefig(FIG / out, dpi=150, bbox_inches='tight')
print('wrote', FIG / out)
