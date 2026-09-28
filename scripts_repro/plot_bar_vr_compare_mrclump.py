"""Does the v_R quadrupole of the low-apocentre high-alpha halo stars match the
Galactic bar, or is it flipped?

Direct same-frame comparison of the mean-v_R field:
  (left)  CONTROL bar tracer: all clean inner stars, 2<R<5 kpc (the canonical bar
          quadrupole in this coordinate frame).
  (right) TEST: in-situ Davies-halo, [Mg/Fe]>0.3, r_apo<5 kpc.
Both in the AstroNN Cartesian frame (Sun toward +x). Rotation sense and the m=2
Fourier phase of mean v_R in the 2<R<4.5 kpc annulus are printed for each, so the
two orientations can be compared quantitatively (same phase -> matches; ~90 deg
apart -> flipped).

Output figures_repro/01_bar_vr_compare_mrclump.png.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
from scipy.stats import binned_statistic_2d
from astropy.io import fits
import matplotlib.pyplot as plt
REPO = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/eos-figures')
sys.path.insert(0, str(REPO))
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
c = Cuts()
FIG = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/figures_repro')
cat = load_catalog('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c); base = np.asarray(m['base'], bool)
feh = np.asarray(cat['fe_h'], float); mg = np.asarray(cat['mg_fe'], float); al = np.asarray(cat['al_fe'], float)
lz = np.asarray(cat['lz'], float); rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float)
ecc = (rap - rperi)/(rap + rperi); aid = np.asarray(cat['apogee_id'])

ann = fits.open('/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE/apogee_astroNN-DR17.fits')[1].data
def norm(a): return np.array([(s.decode() if isinstance(s, bytes) else str(s)).strip() for s in np.asarray(a)])
nid = norm(ann['APOGEE_ID']); o = np.argsort(nid); nid_s = nid[o]
p = np.clip(np.searchsorted(nid_s, aid), 0, len(nid_s)-1); ok = nid_s[p] == aid; src = o[p]
cc = lambda n: np.where(ok, np.asarray(ann[n], float)[src], np.nan)
R, phi, vR, vt = cc('galr'), cc('galphi'), cc('galvr'), cc('galvt')
x = R*np.cos(phi); y = R*np.sin(phi)
# tangential velocity vector (for rotation sense): v_y at the Sun tells the direction
vx = vR*np.cos(phi) - vt*np.sin(phi); vy = vR*np.sin(phi) + vt*np.cos(phi)

halo = base & ((ecc > 0.7) | (lz < 0)) & (al > c.alfe_cut) & (feh > -1.0)
control = base & np.isfinite(R) & (R > 2) & (R < 5)                    # canonical inner bar tracer
test = base & (al > c.alfe_cut) & (feh > 0) & (mg < 0.12) & np.isfinite(R) & (R < 5)  # metal-rich clump, NO halo cut (it is a rotating pop)

# rotation sense: mean v_y for near-Sun disc stars (R~8)
near = base & np.isfinite(R) & (np.abs(R - 8) < 1) & np.isfinite(vy)
print(f'rotation check: mean v_y for R~8 disc = {np.nanmean(vy[near]):+.0f} km/s '
      f'(Sun at +x; disc rotates toward {"+y" if np.nanmean(vy[near])>0 else "-y"})')


def m2_phase(sel, Rlo=2.0, Rhi=4.5):
    s = sel & np.isfinite(x) & np.isfinite(y) & np.isfinite(vR) & (R > Rlo) & (R < Rhi)
    ph = np.arctan2(y[s], x[s])
    c2 = np.mean(vR[s] * np.exp(-2j * ph))          # complex m=2 amplitude of mean v_R
    return np.degrees(0.5 * np.angle(c2)) % 180, 2 * np.abs(c2), int(s.sum())


phc, ampc, nc = m2_phase(control)
pht, ampt, nt = m2_phase(test)
dphi = (pht - phc + 90) % 180 - 90                   # signed phase difference in [-90,90)
print(f'CONTROL bar tracer : m=2 phase = {phc:5.1f} deg, amp = {ampc:4.1f} km/s, n={nc}')
print(f'TEST halo MR-clump : m=2 phase = {pht:5.1f} deg, amp = {ampt:4.1f} km/s, n={nt}')
print(f'phase difference = {dphi:+.1f} deg  -> {"SAME orientation (bar-like)" if abs(dphi)<45 else "FLIPPED (~90 deg off)"}')

EXT = 5.0; NB = 16; NMINB = 5
fig, ax = plt.subplots(1, 2, figsize=(14.5, 6.6), constrained_layout=True)
for a, (lab, sel) in zip(ax, [('CONTROL: inner disc (2<R<5 kpc)', control),
                              (r'TEST: halo [Fe/H]>0, [Mg/Fe]<0.12', test)]):
    mv = binned_statistic_2d(x[sel], y[sel], vR[sel], statistic='mean', bins=NB, range=[(-EXT, EXT), (-EXT, EXT)]).statistic
    cn = binned_statistic_2d(x[sel], y[sel], None, statistic='count', bins=NB, range=[(-EXT, EXT), (-EXT, EXT)]).statistic
    im = a.imshow(np.where(cn >= NMINB, mv, np.nan).T, origin='lower', extent=[-EXT, EXT, -EXT, EXT],
                  aspect='equal', cmap='RdBu_r', vmin=-70, vmax=70, interpolation='nearest')
    for rad in (2, 4.5):
        th = np.linspace(0, 2*np.pi, 120); a.plot(rad*np.cos(th), rad*np.sin(th), color='0.4', lw=0.8, ls='--')
    a.plot(0, 0, '+', color='k', ms=12, mew=2)
    a.annotate('to Sun', xy=(EXT-0.1, 0), xytext=(EXT-0.1, 1.3), ha='right', fontsize=11,
               arrowprops=dict(arrowstyle='->', color='k'))
    a.set_xlim(-EXT, EXT); a.set_ylim(-EXT, EXT); a.set_xlabel('x [kpc]  (+x toward Sun)'); a.set_ylabel('y [kpc]')
    a.set_title(lab, fontsize=13)
    fig.colorbar(im, ax=a, pad=0.02).set_label(r'mean $v_R$ [km/s]')
fig.suptitle(f'bar $v_R$ quadrupole: control vs halo MR-clump   '
             f'(m=2 phase {phc:.0f}$^\\circ$ vs {pht:.0f}$^\\circ$, '
             f'{"MATCH" if abs(dphi)<45 else "FLIPPED"})', fontsize=13)
fig.savefig(FIG / '01_bar_vr_compare_mrclump.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_bar_vr_compare_mrclump.png')
