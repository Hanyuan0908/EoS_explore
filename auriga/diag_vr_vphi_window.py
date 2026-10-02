"""Diagnostic: au18_vr_vphi_three for an ARBITRARY formation window and radial cut.

    usage:  diag_vr_vphi_window.py [T_LO] [T_HI] [R_MIN] [Z_MIN] [ZMAX_MIN]
            (default 4.5 5.5 5 0 0)

Three ways to step outside the bar, any combination, each may be 0:
  R_MIN      cylindrical R at z = 0
  Z_MIN      instantaneous |z| at z = 0
  ZMAX_MIN   z_max, the maximum height the ORBIT reaches, from integrating it in
             the z = 0 AGAMA potential (prep_zmax_z0.py).  The best of the three:
             R keeps bar stars currently at large radius, |z| is a snapshot of
             orbital phase and keeps a bar star caught at the top of its small
             oscillation while discarding a halo star crossing the plane, and
             z_max is a property of the orbit itself.

A WARNING ABOUT Z_MIN.  Cutting |z| > 2 kpc does exclude the bar, which is a
flattened structure, but it is NOT selection-neutral the way R_MIN is: it selects
directly on vertical excursion, which is the quantity most correlated with being
dynamically hot.  It will therefore raise the born-hot fraction for reasons that
have nothing to do with the bar, and the resulting ratio must not be read as a
channel ratio.  R_MIN removes a region; Z_MIN removes a population.

Why this exists rather than a flag on the paper script: the paper figure reads
merger_birth_vs_z0_kinematics.npz, which is frozen to t_form = 4.99-6.54 Gyr and
additionally carries a spherical aperture 3 < r < 30 kpc that removes 49 per cent
of the in-situ stars of that epoch (median r of the removed stars is 1.37 kpc).
Neither can be relaxed from the stored file, so the birth velocities are
remeasured here from the snapshots.

METHOD, matching ana_merger_birth_vs_z0_kinematics exactly:
  * every star is assigned to the FIRST stored snapshot at or after it formed;
  * in that snapshot the galaxy is centred on the main subhalo, the bulk velocity
    of the inner 10 kpc removed, and util.align_galaxy applied, which puts the
    disc angular momentum on component 0 so the disc plane is (1, 2);
  * v_R and v_phi are the cylindrical components in that frame, with the sign of
    v_phi fixed by the median rotation of the 3 < R < 12 kpc, |z| < 2 kpc disc, so
    the disc rotates positive at every epoch.
Present-day v_phi, eccentricity and R come from z0_insitu_catalog.

NO 3 kpc APERTURE IS APPLIED HERE.  The only radial cut is the cylindrical
R > R_MIN at z = 0, which is explicit on the figure.  Counts therefore will not
match CONVENTIONS.md, which quotes numbers that inherit the aperture.

Writes figures/au18_vr_vphi_window_<TLO>_<THI>_R<N>.png and caches the birth
velocities in out/birth_vrvphi_<TLO>_<THI>.npz so a re-plot is instant.
"""
import gc, os, sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod, util

OUT = C.FIG_DIR
os.makedirs(OUT, exist_ok=True)
T_LO = float(sys.argv[1]) if len(sys.argv) > 1 else 4.5
T_HI = float(sys.argv[2]) if len(sys.argv) > 2 else 5.5
R_MIN = float(sys.argv[3]) if len(sys.argv) > 3 else 5.
Z_MIN = float(sys.argv[4]) if len(sys.argv) > 4 else 0.
ZMAX_MIN = float(sys.argv[5]) if len(sys.argv) > 5 else 0.
VPHI_MAX, ECC_MIN, VPHI_SPLIT = 80., 0.6, 150.
RNG = [[-400, 400], [-300, 400]]
NX, NY = 120, 105
CUTC = '#E8112D'
CACHE = C.OUT_DIR + f'/birth_vrvphi_{T_LO:g}_{T_HI:g}.npz'

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 13.5, 'axes.labelsize': 15,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12.5,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 150, 'savefig.dpi': 200,
})
CMAP = ListedColormap(plt.get_cmap('YlGnBu')(np.linspace(.18, 1., 256)))

cat = np.load(C.OUT_DIR + '/z0_insitu_catalog.npz')
st = np.load(C.OUT_DIR + '/snapshot_times.npz')
SN, TS = st['snaps'], st['t_snap']
win = (cat['tform'] >= T_LO) & (cat['tform'] < T_HI)
ids = cat['ids'][win]
print(f'in-situ stars with {T_LO} <= t_form < {T_HI} Gyr: {win.sum():,}')

if os.path.exists(CACHE):
    c = np.load(CACHE)
    assert np.array_equal(c['ids'], ids), 'cache is for a different window'
    bvR, bvphi = c['bvR'], c['bvphi']
    print(f'birth velocities from cache {CACHE}')
else:
    bvR = np.full(len(ids), np.nan)
    bvphi = np.full(len(ids), np.nan)
    tf = cat['tform'][win]
    ks = [k for k in range(1, len(SN)) if TS[k] > T_LO and TS[k - 1] < T_HI]
    print('assigning to snapshots ' + ', '.join(str(int(SN[k])) for k in ks))
    for k in ks:
        sn = int(SN[k])
        m = (tf > TS[k - 1]) & (tf <= TS[k])
        if m.sum() == 0:
            continue
        s = snap_mod.load_snapshot(sn, 4, snappath=C.SIM_DIR, verbose=False,
            loadlist=['ParticleIDs', 'Coordinates', 'Velocities', 'Masses',
                      'GFM_StellarFormationTime'])
        real = s.data['GFM_StellarFormationTime'] > 0
        for key in list(s.data):
            s.data[key] = s.data[key][real]
        sf = sub_mod.subfind(sn, directory=C.SIM_DIR,
                             loadlist=['GroupFirstSub', 'SubhaloPos'])
        cen = sf.data['SubhaloPos'][int(sf.data['GroupFirstSub'][0])]
        util.CentreOnHalo(s, cen)
        rr = np.sqrt((s.data['Coordinates'] ** 2).sum(1))
        q = rr < .01
        bulk = np.average(s.data['Velocities'][q], axis=0,
                          weights=s.data['Masses'][q])
        s.data['Velocities'] -= bulk
        util.align_galaxy(s, radialcut=.01)
        x = s.data['Coordinates'] * 1000.
        v = s.data['Velocities']
        R = np.hypot(x[:, 1], x[:, 2])
        Rs = np.where(R > .1, R, 1.)
        vR = (x[:, 1] * v[:, 1] + x[:, 2] * v[:, 2]) / Rs
        vp = (x[:, 1] * v[:, 2] - x[:, 2] * v[:, 1]) / Rs
        disc = (R > 3) & (R < 12) & (np.abs(x[:, 0]) < 2)
        if np.median(vp[disc]) < 0:
            vp = -vp
        sid = s.data['ParticleIDs']
        del s, x, v; gc.collect()
        o = np.argsort(sid); ss = sid[o]
        want = ids[m]
        p = np.searchsorted(ss, want)
        hit = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == want)
        idx = np.flatnonzero(m)[hit]
        bvR[idx] = vR[o[p[hit]]]
        bvphi[idx] = vp[o[p[hit]]]
        print(f'  snapshot {sn} (t = {TS[k]:.3f}): {m.sum():>7,} stars, '
              f'{hit.sum():>7,} matched', flush=True)
        del sid, vR, vp, R; gc.collect()
    np.savez(CACHE, ids=ids, bvR=bvR, bvphi=bvphi)
    print(f'cached {CACHE}')

zvphi, ecc, Rz0 = cat['vphi'][win], cat['ecc'][win], cat['R'][win]
Zz0 = np.abs(cat['z'][win])
ZMX = np.full(len(ids), np.nan)
if ZMAX_MIN > 0:
    # the z_max file is built once over 4.5-6.5 Gyr and covers every window
    # inside it; a star missing from it is one the integration dropped
    zf = C.OUT_DIR + '/zmax_z0_4.5_6.5.npz'
    if not os.path.exists(zf):
        raise SystemExit(f'missing {zf}; run "prep_zmax_z0.py 4.5 6.5" first')
    zc = np.load(zf)
    oz = np.argsort(zc['ids']); zi = zc['ids'][oz]
    pz = np.searchsorted(zi, ids)
    hz = (pz < len(zi)) & (zi[np.minimum(pz, len(zi) - 1)] == ids)
    ZMX[hz] = zc['zmax'][oz[pz[hz]]]
    print(f'z_max matched for {hz.sum():,} of {len(ids):,} '
          f'({100 * hz.mean():.1f} %)')
zvR = cat['vR'][win]
fin = np.isfinite(bvR) & np.isfinite(bvphi)
keep = fin & (Rz0 > R_MIN) & (Zz0 > Z_MIN)
if ZMAX_MIN > 0:
    keep = keep & np.isfinite(ZMX) & (ZMX > ZMAX_MIN)
eos = keep & (np.abs(zvphi) < VPHI_MAX) & (ecc > ECC_MIN)
hot, cold = eos & (bvphi < VPHI_SPLIT), eos & (bvphi >= VPHI_SPLIT)
print(f'\nbirth velocity measured for {fin.sum():,} of {len(ids):,}')
cutlab = ' and '.join(([f'$R > {R_MIN:g}$ kpc'] if R_MIN > 0 else [])
                      + ([f'$|z| > {Z_MIN:g}$ kpc'] if Z_MIN > 0 else [])
                      + ([rf'$z_{{\rm max}} > {ZMAX_MIN:g}$ kpc']
                         if ZMAX_MIN > 0 else [])) or 'no spatial cut'
print(f'after R > {R_MIN:g}, |z| > {Z_MIN:g}, z_max > {ZMAX_MIN:g} kpc: '
      f'{keep.sum():,} of {int(fin.sum()):,} '
      f'({100 * keep.sum() / fin.sum():.1f} %)')
print(f'  Eos-like  {eos.sum():,}  ({100 * eos.sum() / keep.sum():.2f} % of the parent)')
print(f'  born-hot  {hot.sum():,}  ({100 * hot.sum() / eos.sum():.1f} % of Eos)')
print(f'  born-cold {cold.sum():,}  ({100 * cold.sum() / eos.sum():.1f} % of Eos)')

TIT = [rf'All stars, ${T_LO} < t_{{\rm form}} < {T_HI}$ Gyr, at $z=0$',
       r'Selected Eos-like stars, at $z=0$',
       r'Selected Eos-like stars, at birth']
TIT = [t + f'  ({cutlab})' for t in TIT]
fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.3), sharex=True, sharey=True)
panels = [(zvR, zvphi, keep, True, False, '(a)'),
          (zvR, zvphi, eos, True, False, '(b)'),
          (bvR, bvphi, eos, False, True, '(c)')]
for ax, (x, y, m, band, split, tag), title in zip(axes, panels, TIT):
    h, xe, ye = np.histogram2d(x[m], y[m], bins=[NX, NY], range=RNG)
    h = np.where(h > 0, h / h.max(), np.nan)
    im = ax.pcolormesh(xe, ye, h.T, cmap=CMAP, norm=LogNorm(vmin=1e-3, vmax=1),
                       rasterized=True)
    ax.axhline(0, color='.6', lw=.6, zorder=1)
    ax.axvline(0, color='.6', lw=.6, zorder=1)
    if band:
        ax.axhspan(-VPHI_MAX, VPHI_MAX, color=CUTC, alpha=.07, lw=0, zorder=2)
        for v_ in (-VPHI_MAX, VPHI_MAX):
            ax.axhline(v_, color=CUTC, lw=2.0, ls='--', zorder=3)
    if split:
        ax.axhline(VPHI_SPLIT, color=CUTC, lw=2.4, ls='--', zorder=3)
        bb = dict(fc='white', ec='none', alpha=.88, pad=3.0)
        ax.annotate('born-cold', (385, 350), fontsize=20, ha='right', va='center',
                    color=CUTC, bbox=bb, zorder=4)
        ax.annotate('born-hot', (385, -230), fontsize=20, ha='right', va='center',
                    color=CUTC, bbox=bb, zorder=4)
    ax.text(.035, .955, tag, transform=ax.transAxes, va='top', fontsize=16,
            fontweight='bold')
    ax.text(.035, .055, f'N = {int(m.sum()):,}', transform=ax.transAxes,
            va='bottom', fontsize=13,
            bbox=dict(fc='white', ec='none', alpha=.85, pad=2))
    ax.set_title(title, fontsize=12.5, pad=8)
    ax.set_xlabel(r'$v_R$ [km s$^{-1}$]')
    ax.set(aspect='equal', xlim=(-400, 400), ylim=(-300, 400),
           xticks=np.arange(-400, 401, 200), yticks=np.arange(-300, 401, 100))
axes[0].set_ylabel(r'$v_\phi$ [km s$^{-1}$]')
cb = fig.colorbar(im, ax=axes, fraction=.020, pad=.012)
cb.set_label("density, normalised to each panel's peak")
f = (f'{OUT}/au18_vr_vphi_window_{T_LO:g}_{T_HI:g}'
     f'_R{R_MIN:g}_z{Z_MIN:g}_zmax{ZMAX_MIN:g}.png')
fig.savefig(f, bbox_inches='tight')
print(f'\nsaved {f}')
