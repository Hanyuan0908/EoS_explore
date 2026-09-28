"""Diagnostics for the bimodal r_apo of the in-situ ([Al/Fe]>-0.12) Davies halo,
[Fe/H]>-1:
  (left)  [Mg/Fe]-[Fe/H] scatter coloured by AstroNN age (are the two r_apo modes
          different ages?).
  (right) Galactocentric x-y plane coloured by v_R for the LOW-apocentre
          (r_apo<5 kpc) subset of the HIGH-alpha ([Mg/Fe]>0.3) population -- a bar
          reveals itself as a quadrupole in v_R.

Positions/velocities (galr, galphi, galvr) from the Mac-only AstroNN VAC, matched by
APOGEE_ID. Output figures_repro/01_davies_rapo_diagnose.png.
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
c = Cuts()
FIG = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/figures_repro')
cat = load_catalog('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c); base = np.asarray(m['base'], bool)
feh = np.asarray(cat['fe_h'], float); mg = np.asarray(cat['mg_fe'], float); al = np.asarray(cat['al_fe'], float)
lz = np.asarray(cat['lz'], float); rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float)
age = np.asarray(cat['age'], float); aerr = np.asarray(cat['age_model_error'], float)
ecc = (rap - rperi)/(rap + rperi); aid = np.asarray(cat['apogee_id'])
halo = base & ((ecc > 0.7) | (lz < 0)) & (al > c.alfe_cut) & (feh > -1.0) & np.isfinite(rap)

# positions/velocities from the Mac-only AstroNN VAC
ann = fits.open('/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE/apogee_astroNN-DR17.fits')[1].data
def norm(a): return np.array([(s.decode() if isinstance(s, bytes) else str(s)).strip() for s in np.asarray(a)])
nid = norm(ann['APOGEE_ID']); o = np.argsort(nid); nid_s = nid[o]
p = np.clip(np.searchsorted(nid_s, aid), 0, len(nid_s)-1); ok = nid_s[p] == aid; src = o[p]
cc = lambda n: np.where(ok, np.asarray(ann[n], float)[src], np.nan)
R, phi, vR = cc('galr'), cc('galphi'), cc('galvr')
x = R*np.cos(phi); y = R*np.sin(phi)

XR = (-1.1, 0.6); YR = (-0.1, 0.5)
fig, ax = plt.subplots(1, 2, figsize=(15.5, 6.4), constrained_layout=True)

# ---- left: Mg-Fe scatter coloured by age ----
sA = halo & np.isfinite(age) & np.isfinite(aerr) & (age > 0) & (age < 14) & (aerr/age < 0.3) & np.isfinite(feh) & np.isfinite(mg)
oa = np.argsort(age[sA])                                    # old plotted on top
scA = ax[0].scatter(feh[sA][oa], mg[sA][oa], c=age[sA][oa], s=10, cmap='viridis',
                    vmin=3, vmax=11, edgecolors='none', alpha=0.9, rasterized=True)
ax[0].axhline(0.3, color='k', ls=':', lw=1.4, zorder=2)
ax[0].text(-0.62, 0.16, 'Eos', color='k', fontsize=14, fontweight='bold', zorder=3)
ax[0].set_xlim(*XR); ax[0].set_ylim(*YR); ax[0].set_xlabel('[Fe/H]'); ax[0].set_ylabel('[Mg/Fe]')
ax[0].set_title(rf'in-situ Davies halo, coloured by age (n={int(sA.sum())})')
cbA = fig.colorbar(scA, ax=ax[0], pad=0.02); cbA.set_label('age [Gyr] (AstroNN)')

# ---- right: inner x-y mean v_R map, low-apo high-alpha (look for the bar quadrupole) ----
from scipy.stats import binned_statistic_2d
LOWAPO = 5.0
sB = halo & (mg > 0.3) & (rap < LOWAPO) & np.isfinite(x) & np.isfinite(y) & np.isfinite(vR)
EXT = 5.5; NB = 18; NMINB = 5
mv = binned_statistic_2d(x[sB], y[sB], vR[sB], statistic='mean', bins=NB, range=[(-EXT, EXT), (-EXT, EXT)]).statistic
cn = binned_statistic_2d(x[sB], y[sB], None, statistic='count', bins=NB, range=[(-EXT, EXT), (-EXT, EXT)]).statistic
mvj = np.where(cn >= NMINB, mv, np.nan)
imB = ax[1].imshow(mvj.T, origin='lower', extent=[-EXT, EXT, -EXT, EXT], aspect='equal',
                   cmap='RdBu_r', vmin=-70, vmax=70, interpolation='nearest', zorder=0)
for rad in (3, 5):
    th = np.linspace(0, 2*np.pi, 120)
    ax[1].plot(rad*np.cos(th), rad*np.sin(th), color='0.4', lw=0.8, ls='--', zorder=2)
ax[1].plot(0, 0, '+', color='k', ms=12, mew=2, zorder=3)                        # Galactic centre
ax[1].annotate('to Sun\n(x=+8)', xy=(EXT-0.2, 0), xytext=(EXT-0.2, 1.4), ha='right', fontsize=11,
               arrowprops=dict(arrowstyle='->', color='k'))                     # Sun toward +x (AstroNN)
ax[1].set_xlim(-EXT, EXT); ax[1].set_ylim(-EXT, EXT)
ax[1].set_xlabel('x [kpc]  (+x toward Sun)'); ax[1].set_ylabel('y [kpc]')
ax[1].set_title(rf'high-$\alpha$ ([Mg/Fe]$>0.3$), $r_{{\rm apo}}<{LOWAPO:.0f}$ kpc: mean $v_R$ (n={int(sB.sum())})')
cbB = fig.colorbar(imB, ax=ax[1], pad=0.02); cbB.set_label(r'mean $v_R$ [km/s]')

fig.savefig(FIG / '01_davies_rapo_diagnose.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_davies_rapo_diagnose.png')
print(f'age panel n={int(sA.sum())}; low-apo high-alpha n={int(sB.sum())}')
for lab, sel in [('mg>0.3 & rap<5', halo & (mg > 0.3) & (rap < 5)),
                 ('mg>0.3 & rap>=5', halo & (mg > 0.3) & (rap >= 5))]:
    a = age[sel & np.isfinite(age) & (aerr/age < 0.3)]
    print(f'  {lab}: n={int(sel.sum())}, age med={np.median(a):.1f} (n_age={a.size})')
