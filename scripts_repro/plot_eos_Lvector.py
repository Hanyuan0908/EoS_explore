"""Does Eos rotate about an axis other than z? Compute the full angular-momentum vector
L = r x v (from AstroNN galactocentric R,phi,z, vR,vt,vz) for Eos and controls, and test
whether the mean vector <L> has a significant component off the disc (z) axis.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
REPO = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/eos-figures')
sys.path.insert(0, str(REPO))
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
from eos_figures.plotting import label_axes
rng = np.random.default_rng(0); c = Cuts()
FIG = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/figures_repro')
cat = load_catalog('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c)
feh = np.asarray(cat['fe_h'], float); vphi = np.asarray(cat['galvt'], float); mg = np.asarray(cat['mg_fe'], float)
rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float); lz = np.asarray(cat['lz'], float)
ecc = (rap - rperi) / (rap + rperi); aid = np.asarray(cat['apogee_id'])

ann = fits.open('/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE/apogee_astroNN-DR17.fits')[1].data
def norm(a): return np.array([(s.decode() if isinstance(s, bytes) else str(s)).strip() for s in np.asarray(a)])
nid = norm(ann['APOGEE_ID']); o = np.argsort(nid); nid_s = nid[o]
p = np.clip(np.searchsorted(nid_s, aid), 0, len(nid_s)-1); ok = nid_s[p] == aid; src = o[p]
cc = lambda n: np.where(ok, np.asarray(ann[n], float)[src], np.nan)
R, phi, z = cc('galr'), cc('galphi'), cc('galz'); vR, vt, vz = cc('galvr'), cc('galvt'), cc('galvz')
x = R*np.cos(phi); y = R*np.sin(phi); vx = vR*np.cos(phi)-vt*np.sin(phi); vy = vR*np.sin(phi)+vt*np.cos(phi)
Lx = y*vz - z*vy; Ly = z*vx - x*vz; Lz = x*vy - y*vx
fin = ok & np.isfinite(Lx) & np.isfinite(Ly) & np.isfinite(Lz)

thin_al = np.asarray(m['thin_al'], bool); thick_al = np.asarray(m['thick_al'], bool)
# CANONICAL Eos cut: Davies halo & low-alpha wedge & -0.9<[Fe/H]<-0.2
al = np.asarray(cat['al_fe'], float); base = np.asarray(m['base'], bool)
halo = base & ((ecc > 0.7) | (lz < 0))
def acc(f): return c.slope_acc*f + c.inter_acc
def hl(f): return c.slope_acc2*f + c.inter_acc2
eos = halo & (feh > -0.9) & (feh < -0.2) & (mg > acc(feh)) & (mg < hl(feh)) & (al > c.alfe_cut) & fin
disc = thin_al & (vphi > 150) & (feh > -0.7) & (feh < -0.3) & fin
splash = thick_al & (ecc > 0.7) & (feh > -0.9) & (feh < -0.5) & fin

def stats(s, nb=5000):
    LX, LY, LZ = Lx[s], Ly[s], Lz[s]; n = len(LX)
    mv = np.array([LX.mean(), LY.mean(), LZ.mean()]); sem = np.array([LX.std(), LY.std(), LZ.std()])/np.sqrt(n)
    mags = np.array([np.linalg.norm([LX[i].mean(), LY[i].mean(), LZ[i].mean()])
                     for i in [rng.integers(0, n, n) for _ in range(nb)]])
    return mv, sem, n, mags.mean(), mags.std()

print('mean L vector (kpc km/s):')
for nm, s in [('EOS (ecc>0.7)', eos), ('disc', disc), ('Splash', splash)]:
    mv, sem, n, mmag, smag = stats(s)
    th = np.degrees(np.arccos(mv[2]/np.linalg.norm(mv)))
    print(f'  {nm:14s} n={n:5d}  <Lx>={mv[0]:+6.1f}+-{sem[0]:4.1f}  <Ly>={mv[1]:+6.1f}+-{sem[1]:4.1f}  '
          f'<Lz>={mv[2]:+7.1f}+-{sem[2]:4.1f}  |<L>|={mmag:.0f}+-{smag:.0f}  tilt={th:.0f}deg')
# disc-baseline-subtracted in-plane for Eos
mvE, semE, *_ = stats(eos); mvD, semD, *_ = stats(disc)
dxy = mvE[:2] - mvD[:2]; edxy = np.hypot(semE[:2], semD[:2])
print(f'  Eos in-plane MINUS disc baseline: dLx={dxy[0]:+.1f}+-{edxy[0]:.1f}, dLy={dxy[1]:+.1f}+-{edxy[1]:.1f}')

# figure: TOP row = how Eos is selected; BOTTOM row = Lx, Ly, Lz
base = np.asarray(m['base'], bool)
def bg(a, yv, yr, ylab, sample=None):
    smp = base if sample is None else sample
    s = smp & np.isfinite(feh) & np.isfinite(yv)
    h, xe, ye = np.histogram2d(feh[s], yv[s], bins=[80, 60], range=[(-1.6, 0.4), yr])
    him = np.full_like(h, np.nan); him[h > 0] = np.log10(h[h > 0])
    a.imshow(him.T, origin='lower', extent=[-1.6, 0.4, *yr], aspect='auto', cmap='Greys',
             vmin=np.nanpercentile(him, 3), vmax=np.nanpercentile(him, 99), zorder=0)
    a.set_xlim(-1.6, 0.4); a.set_ylim(*yr); label_axes(a, '[Fe/H]', ylab)

fig, ax = plt.subplots(2, 3, figsize=(15, 8.6), constrained_layout=True)
xx = np.linspace(-1.6, 0.4, 60)
# (a) chemistry: low-alpha wedge + metal-poor
bg(ax[0, 0], mg, (-0.05, 0.45), '[Mg/Fe]')
ax[0, 0].plot(xx, c.slope_acc*xx + c.inter_acc, 'r--', lw=1.1)
ax[0, 0].plot(xx, c.slope_acc2*xx + c.inter_acc2, 'r:', lw=1.4)
ax[0, 0].axvline(-0.9, color='0.4', ls='-', lw=0.8); ax[0, 0].axvline(-0.2, color='0.4', ls='-', lw=0.8)
ax[0, 0].scatter(feh[eos], mg[eos], s=8, c='red', linewidths=0, zorder=3)
ax[0, 0].set_title('(1) chemistry: low-$\\alpha$ in-situ wedge, $-0.9<$[Fe/H]$<-0.2$')
# (b) V_tan (the motivation: ~0) -- background = LOW-ALPHA population only
bg(ax[0, 1], vphi, (-250, 350), r'$V_{\rm tan}$ [km/s]', sample=thin_al)
ax[0, 1].axhline(0, color='k', ls='--', lw=0.8)
ax[0, 1].scatter(feh[eos], vphi[eos], s=8, c='red', linewidths=0, zorder=3)
ax[0, 1].set_title(r'(2) $V_{\rm tan}$ of Eos (net $\approx$ 0);  bg = low-$\alpha$ only')
# (c) eccentricity: halo cut ecc>0.7
bg(ax[0, 2], ecc, (0, 1.02), 'eccentricity')
ax[0, 2].axhline(0.7, color='b', ls='--', lw=1.2)
ax[0, 2].scatter(feh[eos], ecc[eos], s=8, c='red', linewidths=0, zorder=3)
ax[0, 2].set_title('(3) kinematics: $e>0.7$ (halo)')
# bottom: L components
for a, (L, lab) in zip(ax[1], [(Lx[eos], 'Lx'), (Ly[eos], 'Ly'), (Lz[eos], 'Lz')]):
    a.hist(L, bins=np.linspace(-900, 900, 45), color='0.7', edgecolor='0.4')
    mean = L.mean(); sem = L.std()/np.sqrt(len(L))
    a.axvline(0, color='k', lw=1); a.axvspan(mean-sem, mean+sem, color='crimson', alpha=0.3); a.axvline(mean, color='crimson', lw=2)
    a.set_title(f'$\\langle {lab}\\rangle$ = {mean:+.0f} $\\pm$ {sem:.0f}  ({mean/sem:+.1f}$\\sigma$)')
    label_axes(a, f'{lab} [kpc km/s]', 'count' if a is ax[1, 0] else '')
fig.suptitle(f'Eos selection (top; red = {int(eos.sum())} Eos stars) and its angular-momentum vector (bottom): '
             r'weak net $L_z$ only, no significant off-z rotation axis', fontsize=12)
fig.savefig(FIG / '01_eos_Lvector.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_eos_Lvector.png')
